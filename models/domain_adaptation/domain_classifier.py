import torch.nn as nn
from .functional import ReverseLayerF
import torch


class Discriminator(nn.Module):
    """
    A 2-layer MLP for domain classification based on DANN.
    """

    def __init__(self, in_size=512, h=2048, out_size=2, layer=4, conv=False):
        """
        Arguments:
            in_size: size of the input
            h: hidden layer size
            out_size: size of the output
        """

        super().__init__()

        self.h = h
        modules = [nn.Linear(in_size, h)]
        for i in range(layer - 1):
            modules.append(nn.BatchNorm1d(h))
            modules.append(nn.ReLU(True))
            if i == layer-2:
                modules.append(nn.Linear(h, out_size))
            else:
                modules.append(nn.Linear(h, h))
        modules.append(nn.LogSoftmax()) 
        self.net = nn.Sequential(*modules)
        self.out_size = out_size

        # Weight initialization
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, features, alpha):
        """"""
        reverse_features = ReverseLayerF.apply(features, alpha)
        domain_output = self.net(reverse_features)
        non_reversed_output = self.net(ReverseLayerF.apply(features, -alpha))
        return domain_output, non_reversed_output