# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
RelationFormer model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
# from torchvision.ops import nms
import matplotlib.pyplot as plt
import math
import copy

from models.domain_adaptation.domain_classifier import Discriminator

from .deformable_detr_backbone import build_backbone
from .deformable_detr_2D import build_deforamble_transformer
from .utils import nested_tensor_from_tensor_list, NestedTensor, inverse_sigmoid

class RelationFormer(nn.Module):
    """ This is the RelationFormer module that performs object detection """

    def __init__(self, backbone, deformable_transformer, config):
        super().__init__()
        self.encoder = backbone
        self.decoder = deformable_transformer
        self.config = config

        self.num_queries = config.MODEL.DECODER.OBJ_TOKEN + config.MODEL.DECODER.RLN_TOKEN + config.MODEL.DECODER.DUMMY_TOKEN
        self.obj_token = config.MODEL.DECODER.OBJ_TOKEN
        self.hidden_dim = config.MODEL.DECODER.HIDDEN_DIM

        self.num_feature_levels = config.MODEL.DECODER.NUM_FEATURE_LEVELS
        self.two_stage = config.MODEL.DECODER.TWO_STAGE
        self.aux_loss = config.MODEL.DECODER.AUX_LOSS
        self.with_box_refine = config.MODEL.DECODER.WITH_BOX_REFINE
        self.num_classes = config.MODEL.NUM_CLASSES

        self.class_embed = nn.Linear(config.MODEL.DECODER.HIDDEN_DIM, 2)
        self.bbox_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM, config.MODEL.DECODER.HIDDEN_DIM, 4, 3)
        
        if config.MODEL.DECODER.RLN_TOKEN > 0:
            self.relation_embed = MLP(
                config.MODEL.DECODER.HIDDEN_DIM*(2 + config.MODEL.DECODER.RLN_TOKEN),
                config.MODEL.DECODER.HIDDEN_DIM,
                2,
                3
            )
        else:
            self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM*2, config.MODEL.DECODER.HIDDEN_DIM, 2, 3)

        if not self.two_stage:
            self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim*2)    # why *2
        if self.num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = self.encoder.num_channels[i]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.encoder.num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])

        if not config.TRAIN.TRAIN_ENCODER:
            self.input_proj.requires_grad_(False)

        if config.DATA.MIXED:
            self.backbone_domain_discriminator = Discriminator(in_size=self.hidden_dim)
            self.instance_domain_discriminator = Discriminator(in_size=self.hidden_dim*self.num_queries)

        self.decoder.decoder.bbox_embed = None


    def forward(self, samples, seg=True, alpha=1, domain_labels=None):
        if not seg and not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        elif seg:
            samples = nested_tensor_from_tensor_list([tensor.expand(3, -1, -1).contiguous() for tensor in samples])

        # CNN backbone
        features, pos = self.encoder(samples)

        # Create 
        srcs = []
        masks = []
        # For each different input feature level (l=level)
        for l, feat in enumerate(features):
            # Get the feature itself and the corresponding mask
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.encoder[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
            
        if self.config.DATA.MIXED:
            domain_features = []
            replicated_domain_labels = []
            # We have list of 2d features from each feature level where every level has the shape (batch_size, channels, height, width)
            # We want to create a tensor where each position in each feature map is viewed as own sample such that each feature level is (batch_size*height*width, channels) and the domain labels are (batch_size*height*width, 1)
            # This new tensor should be organized in a way such that all feature positions from one sample are grouped together
            # For example, if we have 2 samples in a batch and 2 feature levels, the tensor should look like this:  
            # [sample1_level1, sample1_level2, sample2_level1, sample2_level2]
            for feature in srcs[1:]:
                # For one feature level, put channel dimension last and flatten the tensor
                flat_feature = feature.clone().permute(0,2,3,1)
                # With this operation we get a tensor of shape (batch_size, height*width, channels)
                flat_feature = torch.flatten(flat_feature.clone(), start_dim=1, end_dim=2)
                domain_features.append(flat_feature)
                # Create domain labels by getting a multiplication factor and then replicating each labels from domain_labels to match the number of samples in the feature level to get a tensor of shape (batch_size, height*width)
                domain_label = domain_labels.unsqueeze(1).repeat_interleave(feature.shape[2] * feature.shape[3], dim=1)
                replicated_domain_labels.append(domain_label)

            # Now we merge the list of tensors (shape: (batch_size, height*width, channels)) to one tensor (shape: (batch_size*height*width, channels) such that all feature positions from one sample are grouped together
            conc_features = torch.cat(domain_features, dim=1)
            conc_labels = torch.cat(replicated_domain_labels, dim=1)
            backbone_domain_classifications = self.backbone_domain_discriminator(torch.flatten(conc_features.clone(), end_dim=1), alpha)
            
        else: 
            backbone_domain_classifications = torch.tensor(-1)
            conc_labels = None
    
        hs, init_reference, inter_references, _, _ = self.decoder(
            srcs, masks, query_embeds, pos
        )

        object_token = hs[...,:self.obj_token,:]

        class_prob = self.class_embed(object_token)
        coord_loc = self.bbox_embed(object_token).sigmoid()

        if self.config.DATA.MIXED:
            # Flatten the tensor but keep batch dimension
            domain_hs = torch.flatten(hs.clone(), start_dim=1)
            instance_domain_classifications = self.instance_domain_discriminator(domain_hs, alpha)
        else:
            instance_domain_classifications = torch.tensor(-1)
        
        out = {'pred_logits': class_prob, 'pred_nodes': coord_loc}
        return hs, out, srcs, backbone_domain_classifications, instance_domain_classifications, conc_labels


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


def build_relationformer(config, **kwargs):

    # Backbone consists of actual Backbone followed by positional embedding
    backbone = build_backbone(config)
    deformable_transformer = build_deforamble_transformer(config)

    model = RelationFormer(
        backbone,
        deformable_transformer,
        config,
        **kwargs
    )

    return model
