import os
import time
import torch
import yaml
from medpy.io import load
import pyvista
import json
from argparse import ArgumentParser
import numpy as np

from data.dataset_road_network import build_road_network_data
from models import build_model
from training.inference import relation_infer
from tqdm import tqdm
from functools import partial
from metrics.smd import compute_meanSMD, SinkhornDistance
from metrics.boxap import box_ap, iou_filter, get_unique_iou_thresholds, get_indices_of_iou_for_each_metric, MeanBoxAP, \
    MeanSingleAP
from metrics.box_ops_np import box_iou_np
from metrics.coco import COCOMetric
from monai.data import DataLoader

from utils.utils import image_graph_collate

parser = ArgumentParser()
# TODO the same confg is used for all the models at the moment
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config.')
parser.add_argument('--model',
                    help='Paths to the checkpoints to use for inference separated by a space.')
parser.add_argument('--device', default='cuda',
                    help='device to use for training')
parser.add_argument("--max_samples", default=0, help='On how many samples should the net be evaluated?', type=int)
parser.add_argument('--eval', action='store_true', help='Apply evaluation of metrics')
parser.add_argument('--continuous', action='store_true', help='Continuous rotation of road slice in volume')
parser.add_argument('--general_transformer', action='store_true', help='Run eval on general transformer')

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def main(args):
    """
    Run inference for all the testing data
    """
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['exp_name'])
    config = dict2obj(config)
    device = torch.device("cuda") if args.device == 'cuda' else torch.device("cpu")

    # Build dataloader
    test_ds = build_road_network_data(
        config,
        mode='test',
        debug=False,
        gaussian_augment=True,
        rotate=True,
        continuous=args.continuous,
        max_samples=args.max_samples
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        collate_fn=lambda x: image_graph_collate(x, False, gaussian_augment=True),
        pin_memory=True
    )

    # Build model
    if args.general_transformer:
        net = build_model(config, general_transformer=True, pretrain=True, output_dim=2).to(device)
    else:
        net = build_model(config).to(device)

    # print('Loading model from:', args.model)
    checkpoint = torch.load(args.model, map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    net.eval()  # Put the CNN in evaluation mode

    mixed_ap_metric = MeanBoxAP(
        on_edges=False,
        max_detections=100
    )
    node_ap = MeanSingleAP(mixed_ap_metric, lambda x: None, False)
    edge_ap = MeanSingleAP(mixed_ap_metric, lambda x: None, True)

    for idx, (images, segs, nodes_batch, edges_batch, z_pos) in tqdm(enumerate(test_loader)):
        images = images.to(device, non_blocking=False)
        segs = segs.to(device, non_blocking=False)
        nodes_batch = [node.to(device, non_blocking=False) for node in nodes_batch]
        edges_batch = [edge.to(device, non_blocking=False) for edge in edges_batch]

        h, out = net(images)
        out = relation_infer(h.detach(), out, net, config.MODEL.DECODER.OBJ_TOKEN, config.MODEL.DECODER.RLN_TOKEN)

        if args.eval:
            boxes = [torch.cat([node, 0.2*torch.ones(node.shape, device=device)], dim=-1) for node in nodes_batch]
            boxes_class = [np.zeros(n.shape[0]) for n in boxes]

            mixed_ap_metric.update((
                np.array(out["pred_boxes"]),
                np.array(out["pred_boxes_class"]),
                np.array(out["pred_boxes_score"]),
                np.array([box.to('cpu') for box in boxes]),
                np.array(boxes_class),
                np.array(out["pred_nodes"]),
                np.array(out['pred_rels']),
                np.array(out["pred_rels_class"]),
                np.array(out["pred_rels_score"]),
                nodes_batch,
                edges_batch
            ))

        if idx > 0 and idx % 100 == 0:
            print(f"idx: {idx}, mixed mAP: {mixed_ap_metric.compute()}")
            print(f"idx: {idx}, node mAP: {node_ap.compute()}")
            print(f"idx: {idx}, edge mAP: {edge_ap.compute()}")
            mixed_ap_metric.reset()

        if args.max_samples != 0 and idx >= args.max_samples:
            print("exit..")
            break



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)