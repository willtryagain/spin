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


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def find_patch_len(seq_len, m=1):
    return [
        seq_len // pow(2, m),
        seq_len // pow(2, m + 1),
        seq_len // pow(2, m + 2),
    ]


class MTST(nn.Module):
    def __init__(
        self,
        window: int,
        multiplier: int,
        node_index: int,
        num_heads: int = 2,
        dropout: float = 0.3,
        multipliers: [int] = [16, 12, 8],
        num_encoders: int = 2,
    ):
        super().__init__()
        # print all arguments

        self.node_index = node_index
        self.devices = ["cuda:1", "cuda:2", "cuda:2", "cuda:3", "cuda:3", "cuda:0"]
        self.window = window
        self.multiplier = multiplier
        strides = [1, 1, 1]
        T_S = window
        self.layers = nn.ModuleList([])

        device_index = 0
        self.index_cnt = -2

        for i in range(len(multipliers) - 1):
            self.layers.append(
                MTST_layer(
                    multipliers[i] * T_S,
                    num_heads,
                    num_encoders,
                    dropout,
                    strides,
                    [2 * T_S, 4 * T_S, 8 * T_S],
                    self.devices[device_index],
                )
            )
            device_index += 1
            self.layers.append(
                nn.Linear(
                    multipliers[i] * T_S,
                    multipliers[i + 1] * T_S,
                    device=self.devices[device_index],
                )
            )
            device_index += 1
            self.layers.append(nn.SELU())
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(
                BatchNorm1d(
                    multipliers[i + 1] * T_S, device=self.devices[device_index - 1]
                )
            )

        self.layers.append(
            MTST_layer(
                multipliers[-1] * T_S,
                num_heads,
                num_encoders,
                dropout,
                strides,
                [2 * T_S, 4 * T_S, 8 * T_S],
                self.devices[device_index],
            )
        )

        device_index += 1
        self.layers.append(
            nn.Linear(
                multipliers[-1] * T_S,
                multipliers[0] * T_S,
                device=self.devices[device_index],
            )
        )
        self.prev_log = None
        self.missing_emb = nn.Parameter(torch.randn(2, 1), requires_grad=True)
        self.valid_emb = nn.Parameter(torch.randn(2, 1), requires_grad=True)

        self.apply(init_weights)

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
        x = x.squeeze()
        self.index_cnt += 1
        cur_max_log = []
        assert not torch.isnan(x).any()

        mask = mask.squeeze()
        B, L, n = x.shape
        x = x.permute((1, 0, 2)).flatten(start_dim=1).permute((1, 0))
        mask = mask.permute((1, 0, 2)).flatten(start_dim=1).permute((1, 0))
        # x - [B x L]

        # Whiten missing values
        h = x * mask
        h = torch.where(mask.bool(), h + self.valid_emb[0], h + self.missing_emb[0])
        device_index = 0
        for i, layer in enumerate(self.layers):
            if layer.__class__.__qualname__ in ["MTST_layer", "Linear"]:
                h = h.to(self.devices[device_index])
                layer = layer.to(self.devices[device_index])
                prev = h
                h = layer(h)
                if torch.isnan(h).any() and not torch.isnan(prev).any():
                    ic(i, layer.__class__.__qualname__)
                if h.max().item() == torch.inf and prev.max().item() != torch.inf:
                    ic(i, layer.__class__.__qualname__, h.max(), prev.max())
                    ic(cur_max_log)
                    ic(self.prev_log)
                if i == 0:
                    h = torch.where(
                        mask.bool().to(self.devices[device_index]),
                        h + self.valid_emb[1].to(self.devices[device_index]),
                        h + self.missing_emb[1].to(self.devices[device_index]),
                    )
                device_index += 1
            else:
                if layer.__class__.__qualname__ == "BatchNorm1d":
                    layer = layer.to(self.devices[device_index - 1])
                h = layer(h)
            cur_max_log.append((i, h.max().item()))

        h = h.unflatten(0, (B, n)).permute((0, 2, 1))
        h = h.unsqueeze(3)
        self.prev_log = cur_max_log
        return h.to(x.device)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--window", type=int)
        parser.add_argument("--multiplier", type=int)
        parser.add_argument("--node-index", type=int)
        parser.add_argument("--num-encoders", type=int, default=3)
        parser.add_argument("--num-heads", type=int, default=1)
        parser.add_argument("--dropout", type=float, default=0.3)

        return parse_
