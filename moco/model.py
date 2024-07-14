# Adapted MoCo-v3 implementation

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of https://github.com/facebookresearch/moco-v3.

import torch
import torch.nn as nn
import sys

from models import build_model


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, config, device, dim=256, mlp_dim=4096, T=0.2):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        self.device = device
        self.config = config

        # build encoders
        self.base_encoder = build_model(config).to(device)
        self.momentum_encoder = build_model(config).to(device)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        for param_b, param_m in zip(self.base_projector.parameters(), self.momentum_projector.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.config.MODEL.DECODER.HIDDEN_DIM * (self.config.MODEL.DECODER.OBJ_TOKEN + self.config.MODEL.DECODER.RLN_TOKEN)

        # projector
        self.base_projector = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_projector = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)

    @torch.no_grad()
    def update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

        for param_b, param_m in zip(self.base_projector.parameters(), self.momentum_projector.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k, i):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)

        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * i).to(self.device)
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)
    
    def compute_ks(self, x1, x2, m):
        with torch.no_grad():  # no gradient
            # compute momentum features as targets
            k1 = self.momentum_projector(torch.flatten(self.momentum_encoder(x1)[0], start_dim=1))
            k2 = self.momentum_projector(torch.flatten(self.momentum_encoder(x2)[0], start_dim=1))

        return k1, k2

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        out1 = torch.flatten(self.base_encoder(x1)[0], start_dim=1)
        out2 = torch.flatten(self.base_encoder(x2)[0], start_dim=1)

        # Pass through projector
        out1 = self.base_projector(out1)
        out2 = self.base_projector(out2)

        # Pass through Predictor
        q1 = self.predictor(out1)
        q2 = self.predictor(out2)

        return q1, q2
