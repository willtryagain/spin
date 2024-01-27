import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class RelativeGlobalAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError("incompatible `d_model` and `num_heads`")
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        self.qk_coeff = nn.Parameter(torch.tensor(1 / max_len).to(torch.float16))
        self.register_buffer(
            "mask", torch.ones(max_len, max_len).unsqueeze(0).unsqueeze(0)
        )
        # self.mask.shape = (1, 1, max_len, max_len)

        self.apply(init_weights)


    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        if x.max().item() == torch.inf:
            ic("inf x", self.__class__.__qualname__)
        if torch.isnan(x).any():
            ic("nan x", self.__class__.__qualname__)
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError("sequence length exceeds model capacity")

        k_t = (
            self.key(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .permute(0, 2, 3, 1)
        )
        k_t_norm = torch.norm(k_t, dim=-2, keepdim=True) + 1e-5
        k_t = k_t / k_t_norm
        if torch.isnan(k_t).any() and not torch.isnan(x).any():
            ic("k_t", self.__class__.__qualname__)

        if k_t.max().item() == torch.inf:
            ic("inf k_t", self.__class__.__qualname__)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = (
            self.value(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        if v.max().item() == torch.inf:
            ic("inf v", self.__class__.__qualname__)

        if torch.isnan(v).any() and not torch.isnan(x).any():
            ic("v", self.__class__.__qualname__, torch.isnan(x).any())
            ic(x.max(), x.min())
            ic(self.value.weight.isnan().any(), self.value.weight.isinf().any())
            ic(self.value.weight.max(), self.value.weight.min())
            ic(self.value.bias.isnan().any(), self.value.bias.isinf().any())
            ic(self.value.bias.max(), self.value.bias.min())
            ic(self.value.weight.shape, self.value.bias.shape)
        # v.shape = (batch_size, num_heads, seq_len, d_head)

        q = (
            self.query(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        q_norm = torch.norm(q, dim=2, keepdim=True) + 1e-5
        q = q / q_norm

        if q.max().item() == torch.inf:
            ic("inf q", self.__class__.__qualname__)
        if torch.isnan(q).any() and not torch.isnan(x).any():
            ic("q", self.__class__.__qualname__)
        # shape = (batch_size, num_heads, seq_len, d_head)

        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        if torch.isnan(Er_t).any():
            ic("Er_t", self.__class__.__qualname__)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        if (
            torch.isnan(QEr).any()
            and not torch.isnan(q).any()
            and not torch.isnan(Er_t).any()
        ):
            ic("QEr", self.__class__.__qualname__)
        if QEr.max().item() == torch.inf:
            ic("inf QEr", self.__class__.__qualname__)

        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        if torch.isnan(Srel).any() and not torch.isnan(QEr).any():
            ic("Srel", self.__class__.__qualname__)

        if Srel.max().item() == torch.inf:
            ic("inf Srel", self.__class__.__qualname__)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)

        # QK_t = torch.matmul(q.long(), k_t.long())
        QK_t = torch.matmul(q, k_t)

        if (
            torch.isnan(QK_t).any()
            and not torch.isnan(q).any()
            and not torch.isnan(k_t).any()
        ):
            ic("QK_t", self.__class__.__qualname__)

        if QK_t.max().item() == torch.inf:
            ic("inf QK_t", self.__class__.__qualname__)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) * self.qk_coeff
        attn = attn.to(x.dtype)
        if (
            torch.isnan(attn).any()
            and not torch.isnan(QK_t).any()
            and not torch.isnan(Srel).any()
        ):
            ic("attn", self.__class__.__qualname__)

        if attn.max().item() == torch.inf:
            ic("inf attn", self.__class__.__qualname__)
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        prev = attn
        attn = F.softmax(attn, dim=-1)
        if torch.isnan(attn).any() and not torch.isnan(prev).any():
            ic("attn", self.__class__.__qualname__)

        if attn.max().item() == torch.inf:
            ic("inf attn", self.__class__.__qualname__)
        out = torch.matmul(attn, v)
        if (
            torch.isnan(out).any()
            and not torch.isnan(attn).any()
            and not torch.isnan(v).any()
        ):
            ic("out", self.__class__.__qualname__)

        if out.max().item() == torch.inf:
            ic("inf out", self.__class__.__qualname__)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)

        return self.dropout(out).to(torch.float16)

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel
