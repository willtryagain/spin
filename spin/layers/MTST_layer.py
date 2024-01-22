import numpy as np
import torch
import torch.nn as nn
from icecream import ic
from torch.nn import Linear

from .relative_global_attention import RelativeGlobalAttention
from .vanilla_transformer import make_model


def find_num_patches(window, patch_size, stride):
    return max(1, (window - patch_size) // stride + 2)


def create_patch(y, patch_size, stride):
    # [bs x seq_len]
    y_next = y.clone()
    # append the last column stride times
    y_next = torch.cat([y_next, y[:, -1].unsqueeze(1).repeat(1, stride)], dim=1)
    if y_next.shape[1] < patch_size:
        y_next = torch.cat(
            [y_next, y[:, -1].unsqueeze(1).repeat(1, patch_size - y_next.shape[1])],
            dim=1,
        )
    # ic(patch_size * stride, patch_size, stride)

    # split into patches
    y_next = y_next.unfold(1, patch_size, stride).to(y.device)
    return y_next  # [bs  x num_patch  x patch_len]


def find_smallest_divisble_num(num, divisor):
    return (num - (num % divisor)) + divisor


class MTST_layer(nn.Module):
    def __init__(
        self, input_size, num_heads, num_encoders, dropout, strides, patch_sizes, device
    ):
        super().__init__()

        patch_sizes = [
            find_smallest_divisble_num(patch_size, num_heads)
            for patch_size in patch_sizes
        ]
        num_patches = [
            find_num_patches(input_size, patch_sizes[i], strides[i])
            for i in range(len(patch_sizes))
        ]

        self.trans_layers = [
            make_model(
                seq_len,
                num_encoders,
                patch_size,
                12,
                num_heads,
                dropout,
                RelativeGlobalAttention,
                device,
            )
            for (seq_len, patch_size) in zip(num_patches, patch_sizes)
        ]
        patch_sizes = np.array(patch_sizes)
        num_patches = np.array(num_patches)
        strides = np.array(strides)
        flatten_size = (patch_sizes * num_patches).sum()
        self.device = device
        self.ff = Linear(flatten_size, input_size, device=device)
        self.patch_sizes = patch_sizes
        self.input_size = input_size
        self.num_patches = num_patches
        self.strides = strides

    def forward(self, y):
        outputs = []
        for i in range(len(self.patch_sizes)):
            y_i = create_patch(y, self.patch_sizes[i], self.strides[i])
            
            # [bs x num_patch x patch_len]
            # y_i = y_i.permute(0, 2, 1)
            prev = y_i
            y_i = self.trans_layers[i](y_i)
            if torch.isnan(y_i).any() and not torch.isnan(prev).any():
                ic("patch is nan")
            y_i = y_i.flatten(start_dim=1)
            outputs.append(y_i)
            # flatten the dims except first
        outputs = torch.column_stack(outputs)

        self.ff = self.ff.to(self.device)
        y = self.ff(outputs)

        if torch.isnan(y).any() and not torch.isnan(outputs).any():
            ic("final layer ff is nan")
        return y
