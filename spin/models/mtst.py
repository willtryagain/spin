import copy
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from torch import Tensor, nn
from torch.distributed.fsdp.wrap import wrap
from torch.nn import BatchNorm1d, LayerNorm, Linear
from torch_geometric.nn import inits
from torch_geometric.typing import List, OptTensor, Union
from tsl.nn.blocks.encoders import MLP
from tsl.nn.layers import PositionalEncoding


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = BatchNorm1d(layer.size)

    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x)

        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = BatchNorm1d(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x.transpose(1, 2)).transpose(1, 2)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, attn, seq_len, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linear = nn.Linear(d_model, d_model)
        self.attn = attn
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        "Implements Figure 2"
        x = self.attn(x)
        return self.linear(x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


def find_num_patches(window, patch_size, stride):
    return (window - patch_size) // stride + 2


def make_model(
    seq_len,
    N=3,
    d_model=128,
    d_ff=512,
    h=16,
    dropout=0.2,
    device="cpu",
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = RelativeGlobalAttention(d_model, h, seq_len, dropout)

    attn = MultiHeadedAttention(h, d_model, attn, seq_len, dropout).to(device)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(device)
    model = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N).to(device)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def create_patch(y, patch_size, stride):
    # [bs x seq_len]
    y_next = y.clone()
    # append the last column stride times
    y_next = torch.cat([y_next, y[:, -1].unsqueeze(1).repeat(1, stride)], dim=1)
    # split into patches
    y_next = y_next.unfold(1, patch_size, stride).to(y.device)
    return y_next  # [bs  x num_patch  x patch_len]


class MTST_layer(nn.Module):
    def __init__(self, patch_sizes, num_patches, strides, output_size, device):
        super().__init__()
        self.trans_layers = [
            make_model(seq_len=seq_len, d_model=patch_size, device=device)
            for (seq_len, patch_size) in zip(num_patches, patch_sizes)
        ]
        patch_sizes = np.array(patch_sizes)
        num_patches = np.array(num_patches)
        strides = np.array(strides)
        flatten_size = (patch_sizes * num_patches).sum()
        self.ff = Linear(flatten_size, output_size).to(device)
        self.patch_sizes = patch_sizes
        self.output_size = output_size
        self.num_patches = num_patches
        self.strides = strides

    def forward(self, y):
        outputs = []
        for i in range(len(self.patch_sizes)):
            y_i = create_patch(y, self.patch_sizes[i], self.strides[i])
            # [bs x num_patch x patch_len]
            y_i = self.trans_layers[i](y_i)
            y_i = y_i.flatten(start_dim=1)
            outputs.append(y_i)
            # flatten the dims except first
        outputs = torch.column_stack(outputs)
        y = self.ff(outputs)
        return y


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
        self.register_buffer(
            "mask", torch.ones(max_len, max_len).unsqueeze(0).unsqueeze(0)
        )
        # self.mask.shape = (1, 1, max_len, max_len)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError("sequence length exceeds model capacity")

        k_t = (
            self.key(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .permute(0, 2, 3, 1)
        )
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = (
            self.value(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        q = (
            self.query(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        # shape = (batch_size, num_heads, seq_len, d_head)

        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)

        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out)

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


class MTST(nn.Module):
    def __init__(
        self,
        input_size: int,
        seq_len: int,
        node_index: int,
        patch_sizes: List[int] = [16, 32],
        strides: List[int] = [1, 1],
        num_heads: int = 16,
        dropout: float = 0.2,
        hidden_size: int = 512,
        n_layers: int = 1,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.node_index = node_index
        self.device = "cuda"

        self.mask_emb = nn.Embedding(1, 1)
        num_patches = [
            find_num_patches(seq_len, patch_sizes[i], strides[i])
            for i in range(len(patch_sizes))
        ]
        self.layers = nn.ModuleList(
            [
                MTST_layer(
                    patch_sizes, num_patches, strides, seq_len, device=self.device
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        u: Tensor,
        mask: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        node_index: OptTensor = None,
        target_nodes: OptTensor = None,
    ):
        x = x.squeeze(-1)[..., self.node_index]
        mask = mask.squeeze(-1)[..., self.node_index]

        # Whiten missing values
        x = x * mask

        h = torch.where(mask.bool(), x, x + self.mask_emb.weight)
        for layer in self.layers:
            h = layer(h)

        return h

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--seq-len", type=int)
        parser.add_argument("--node-index", type=int)

        return parser


if __name__ == "__main__":
    batch_size = 10
    seq_len = 512
    model = make_model(seq_len)

    patch_sizes = [16]
    strides = [1]
    num_patches = [
        find_num_patches(seq_len, patch_sizes[i], strides[i])
        for i in range(len(patch_sizes))
    ]

    patch_sizes = np.array(patch_sizes)
    num_patches = np.array(num_patches)
    strides = np.array(strides)
    output_size = seq_len

    d_model = 128
    y = torch.rand(batch_size, seq_len, d_model)
    y = model(y)

    print(y.shape)
