import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, cell_size, dim):
        super().__init__()
        self.body = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=cell_size, stride=cell_size)

    def forward(self, x):
        return self.body(x)

