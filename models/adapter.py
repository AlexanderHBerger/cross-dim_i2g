import torch
from torch import nn
from einops import rearrange


class ProjectionAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.mean(x, dim=-1)

class InflationAdapter(nn.Module):
    def __init__(self, depth, dim):
        super().__init__()
        self.collapse = nn.Conv2d(dim * depth, dim, (1, 1))

    def forward(self, x):
        slices = [x[:, :, :, :, i] for i in range(x.shape[-1])]
        x = torch.cat(slices, dim=1)
        return self.collapse(x)

class GroupAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.collapse = nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True)

    def forward(self, x):
        patch_size = x.shape[-1]
        x = rearrange(x, 'b c px py pz -> (b px py) pz c')
        x = self.collapse(x)
        x = x[:, 0]
        x = rearrange(x, '(b px py) c -> b c px py', px=patch_size, py=patch_size)
        return x
