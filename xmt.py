import torch
import torch.nn as nn

device = "cuda:1"
ff = nn.Linear(100, 100, device=device)
print(ff.weight.device)

ff = nn.Linear(100, 100).to(device)
print(ff.weight.device)