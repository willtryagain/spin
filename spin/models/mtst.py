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

from ..layers import MTST_layer


def find_patch_len(seq_len, m=1):
    return [
        seq_len // pow(2, m),
        seq_len // pow(2, m + 1),
        seq_len // pow(2, m + 2),
    ]


class MTST(nn.Module):
    def __init__(
        self,
        seq_len: int,
        node_index: int,
        num_heads: int = 2,
        dropout: float = 0.3,
        num_mtst_ff_layers: int = 8,
        num_encoders: int = 2,
    ):
        super().__init__()
        # print all arguments

        self.num_mtst_ff_layers = num_mtst_ff_layers
        self.node_index = node_index
        self.device = "cuda"
        strides = [1, 2, 2]
        gamma = 0.5

        self.layers = nn.ModuleList(
            [
                MTST_layer(
                    seq_len // pow(2, 0),
                    num_heads,
                    num_encoders,
                    dropout,
                    strides,
                    find_patch_len(seq_len, m=2),
                    self.device,
                )
            ]
        )

        prev = seq_len

        for i in range(num_mtst_ff_layers):
            next = int(prev * (1 - pow(gamma, i + 1)))
            self.layers.append(nn.GELU())
            self.layers.append(
                MTST_layer(
                    prev,
                    num_heads,
                    num_encoders,
                    dropout,
                    strides,
                    find_patch_len(prev, m=2),
                    self.device,
                )
            )
            self.layers.append(nn.GELU())
            self.layers.append(
                nn.Linear(prev, next)
            )
            prev = next
        self.layers.append(nn.GELU())
        self.layers.append(nn.Linear(next, seq_len))

        self.mask_emb = nn.Embedding(2, 1)

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
        # x - [B x L]

        # Whiten missing values
        h = x * mask
        h = torch.where(
            mask.bool(), h, h + self.mask_emb(torch.LongTensor([0]).cuda())[0]
        )
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i == 0:
                h = h + self.mask_emb(torch.LongTensor([1]).cuda())[0]

        return h

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--seq-len", type=int)
        parser.add_argument("--node-index", type=int)
        parser.add_argument("--num-mtst-ff-layers", type=int, default=2)
        parser.add_argument("--num-encoders", type=int, default=3)

        return parser

# Shreyas@1234
# shreyas@1234
# Shreyas1234
# shreyas1234
    
