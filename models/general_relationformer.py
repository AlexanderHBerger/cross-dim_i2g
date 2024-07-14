# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
RelationFormer model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import copy

from .deformable_detr_backbone import build_backbone
from .models2D.deformable_detr_2D import build_deforamble_transformer
from .adapter import ProjectionAdapter, InflationAdapter, GroupAdapter
from .patch_embedding import PatchEmbedding
from .seresnet import build_seresnet
from .utils import nested_tensor_from_tensor_list


class RelationFormer(nn.Module):
    """ This is the RelationFormer module that performs object detection """

    def __init__(self, backbone, deformable_transformer, config, output_dim=3, pretrain=False):
        super().__init__()
        self.encoder = backbone
        self.decoder = deformable_transformer
        self.config = config

        self.num_queries = config.MODEL.DECODER.OBJ_TOKEN + config.MODEL.DECODER.RLN_TOKEN + config.MODEL.DECODER.DUMMY_TOKEN
        self.obj_token = config.MODEL.DECODER.OBJ_TOKEN
        self.hidden_dim = config.MODEL.DECODER.HIDDEN_DIM

        self.num_classes = config.MODEL.NUM_CLASSES

        self.class_embed = nn.Linear(config.MODEL.DECODER.HIDDEN_DIM, 2)
        self.bbox_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM, config.MODEL.DECODER.HIDDEN_DIM, output_dim*2, 3)

        if config.MODEL.DECODER.RLN_TOKEN > 0:
            self.relation_embed = MLP(
                config.MODEL.DECODER.HIDDEN_DIM * (2 + config.MODEL.DECODER.RLN_TOKEN),
                config.MODEL.DECODER.HIDDEN_DIM,
                2,
                3
            )
        else:
            self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM * 2, config.MODEL.DECODER.HIDDEN_DIM, 2, 3)

        if not pretrain and config.MODEL.ADAPTER == "projection":
            self.adapter = ProjectionAdapter()
        elif not pretrain and config.MODEL.ADAPTER == "inflation":
            self.adapter = InflationAdapter(2, config.MODEL.DECODER.HIDDEN_DIM)
        elif not pretrain and config.MODEL.ADAPTER == "group":
            self.adapter = GroupAdapter(config.MODEL.DECODER.HIDDEN_DIM)

        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim * 2)

        self.decoder.decoder.bbox_embed = None
        self.pretrain = pretrain
        self.backbone_type = config.MODEL.ENCODER.TYPE
        if pretrain and self.backbone_type != "EMBEDDING":
            self.input_proj_2d = nn.Sequential(
                    nn.Conv2d(self.encoder.num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )
            self.positional_embedding = nn.Parameter(torch.zeros(1, self.hidden_dim, 2, 2), requires_grad=True)
            print("WARNING: no input proj")
        elif self.backbone_type != "EMBEDDING":
            self.input_proj = nn.Conv3d(self.encoder.num_features, self.hidden_dim, kernel_size=1)
            if not config.TRAIN.TRAIN_ENCODER:
                self.input_proj.requires_grad_(False)
            self.positional_embedding = nn.Parameter(torch.zeros(1, self.hidden_dim, 2, 2), requires_grad=True)
        else:
            self.positional_embedding = nn.Parameter(torch.zeros(1, self.hidden_dim, config.MODEL.ENCODER.CELL_SIZE, 8), requires_grad=True)

    def forward(self, samples, z_pos=None):
        if self.backbone_type == "EMBEDDING":
            features = self.encoder(samples)
            src = features
            mask = torch.ones(src.shape[0], src.shape[2], src.shape[3], dtype=torch.bool, device=src.device)
            srcs = [src]
            mask_list = [mask]
        elif self.pretrain:
            nested_samples = nested_tensor_from_tensor_list([tensor.expand(3, -1, -1).contiguous() for tensor in samples])
            #The positional encoding can be ignored because we will use 3D encoding
            features, _pos = self.encoder(nested_samples)
            src, mask = features[-1].decompose()
            srcs = [self.input_proj_2d(src)]
            mask_list = [mask]
        else:
            # 3D part
            feat_list = [self.encoder(samples)]
            mask_list = [torch.zeros(feat_list[0][:, 0, 0, ...].shape, dtype=torch.bool).to(feat_list[0].device)]
            srcs = self.input_proj(feat_list[-1])

            srcs = [self.adapter(srcs)]

        #print(f"after adapter: {srcs[0].shape}")

        pos = [self.positional_embedding.repeat(samples.shape[0], 1, 1, 1)]

        #print(f"positional: {pos[0].shape}")

        query_embeds = self.query_embed.weight

        hs, init_reference, inter_references, _, _ = self.decoder(
            srcs, mask_list, query_embeds, pos
        )

        object_token = hs[..., :self.obj_token, :]

        class_prob = self.class_embed(object_token)
        coord_loc = self.bbox_embed(object_token).sigmoid()

        out = {'pred_logits': class_prob, 'pred_nodes': coord_loc}
        return hs, out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_general_relationformer(config, pretrain=False, **kwargs):
    # Backbone consists of actual Backbone followed by positional embedding
    if config.MODEL.ENCODER.TYPE == "EMBEDDING":
        backbone = PatchEmbedding(cell_size=config.MODEL.ENCODER.CELL_SIZE, dim=config.MODEL.DECODER.HIDDEN_DIM)
    elif pretrain:
        backbone = build_backbone(config)
    else:
        backbone = build_seresnet(config)

    deformable_transformer = build_deforamble_transformer(config)

    model = RelationFormer(
        backbone,
        deformable_transformer,
        config,
        pretrain=pretrain,
        **kwargs
    )

    return model
