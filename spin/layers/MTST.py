import nn from torch
import torch
from nn import TransformerEncoderLayer, Linear
class MTST_layer(nn.Module):
    def __init__(self, patch_sizes, num_patches, window_size, n_head):
        self.trans_layers = [
            TransformerEncoderLayer(patch_size, n_head) for patch_size in patch_sizes
        ]
        flatten_size = (patch_sizes * num_patches).sum()
        self.ff = Linear(flatten_size, window_size)



    def forward(self, y):
        
        # batch  reshape

        # batch reshape

        pass


def create_patch(y, patch_size, stride):
    # [bs x seq_len x n_vars]
    y_next = y.clone()
    # pad strides times the last value to y
    y_next = torch.cat([y_next, y[:, -1, :].unsqueeze(1).repeat(1, stride, 1)], dim=1)
    # split into patches
    y_next = y_next.unfold(1, patch_size, stride).to(y.device)
    return y_next # [bs x num_patch x n_vars x patch_len]