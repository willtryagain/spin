import numpy as np
import torch
import torch.nn as nn
from icecream import ic
from torch.nn import Linear

from .relative_global_attention import RelativeGlobalAttention
from .vanilla_transformer import make_model


def find_num_patches(window, patch_size, stride):
    return (window - patch_size) // stride + 2


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
                attn=RelativeGlobalAttention,
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
