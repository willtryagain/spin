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

from ..layers import RelativeGlobalAttention


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
    d_ff=336,
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


def find_smallest_divisble_num(num, divisor):
    return num - (num % divisor)


class MTST_layer(nn.Module):
    def __init__(self, output_size, num_heads, N, dropout, device):
        super().__init__()
        strides = [1, 4, 8]
        patch_sizes = [output_size // 4, output_size // 8, output_size // 16]
        patch_sizes = [
            find_smallest_divisble_num(patch_size, num_heads)
            for patch_size in patch_sizes
        ]
        num_patches = [
            find_num_patches(output_size, patch_sizes[i], strides[i])
            for i in range(len(patch_sizes))
        ]
        ic(num_patches)
        self.trans_layers = [
            make_model(
                N=N,
                seq_len=seq_len,
                d_model=patch_size,
                h=num_heads,
                d_ff=patch_size // 4,
                dropout=dropout,
                device=device,
            )
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


class MTST(nn.Module):
    def __init__(
        self,
        input_size: int,
        seq_len: int,
        node_index: int,
        patch_sizes: List[int] = [8, 32, 96],
        strides: List[int] = [12, 8, 4],
        num_heads: int = 2,
        dropout: float = 0.3,
        n_layers: int = 8,
        num_encoders: int = 2,
    ):
        super().__init__()
        # print all arguments

        self.n_layers = n_layers
        self.node_index = node_index
        self.device = "cuda"

        self.mask_emb = nn.Embedding(n_layers, 1)

        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            self.layers.append(
                MTST_layer(
                    seq_len // pow(2, i),
                    num_heads,
                    num_encoders,
                    dropout,
                    self.device,
                )
            )
            self.layers.append(
                nn.Linear(seq_len // pow(2, i), seq_len // pow(2, i + 1))
            )
        self.layers.append(nn.Linear(seq_len // pow(2, n_layers), seq_len))

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
        ic(x.shape)
        x = x.squeeze(-1)[..., self.node_index]
        mask = mask.squeeze(-1)[..., self.node_index]
        # x - [B x L]
        ic(x.shape)

        # Whiten missing values
        x = x * mask

        for i, layer in enumerate(self.layers):
            h = torch.where(
                mask.bool(), x, x + self.mask_emb(torch.LongTensor([i]).cuda())[0]
            )

            h = layer(h)
            ic(h.shape)

        return h

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--seq-len", type=int)
        parser.add_argument("--node-index", type=int)
        parser.add_argument("--n-layers", type=int, default=8)
        parser.add_argument("--num-encoders", type=int, default=3)

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
