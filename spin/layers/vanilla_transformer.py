import copy
import math

import torch
import torch.nn as nn
from icecream import ic
from torch.nn import BatchNorm1d


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



def make_model(
    seq_len,
    N=3,
    d_model=128,
    d_ff=336,
    h=16,
    dropout=0.2,
    attn=None,
    device="cpu",
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = attn(d_model, h, seq_len, dropout)

    attn = MultiHeadedAttention(h, d_model, attn, seq_len, dropout).to(device)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(device)
    model = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N).to(device)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
