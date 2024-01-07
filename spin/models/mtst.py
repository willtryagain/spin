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



