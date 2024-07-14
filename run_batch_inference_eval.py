import os
import time
import torch
import yaml
from medpy.io import load
import pyvista
import json
from argparse import ArgumentParser
import numpy as np
from data.dataset_vessel3d import build_vessel_data
from models import build_model
from training.inference import relation_infer
from tqdm import tqdm
from functools import partial
from metrics.smd import MeanSMD, compute_meanSMD, SinkhornDistance
from metrics.boxap import MeanSingleAP, box_ap, iou_filter, get_unique_iou_thresholds, get_indices_of_iou_for_each_metric, MeanBoxAP
from metrics.box_ops_np import box_iou_np
from metrics.coco import COCOMetric
import networkx as nx
from monai.data import DataLoader

from utils.utils import image_graph_collate

parser = ArgumentParser()
#TODO the same confg is used for all the models at the moment
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
parser.add_argument('--seg', action='store_true', help='Use segs instead of raw images')
parser.add_argument('--general_transformer', action='store_true', help='Run eval on general transformer')
parser.add_argument('--no_strict_loading', default=False, action="store_true",
                    help="Whether the model was pretrained with domain adversarial. If true, the checkpoint will be loaded with strict=false")


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
    device = torch.device("cuda") if args.device=='cuda' else torch.device("cpu")

    nifti_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'raw')
    seg_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'seg')
    vtk_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'vtp')
    nifti_files = []
    vtk_files = []
    seg_files = []

    for file_ in os.listdir(nifti_folder):
        file_ = file_[:-7]
        nifti_files.append(os.path.join(nifti_folder, file_+'.nii.gz'))
        seg_files.append(os.path.join(seg_folder, file_[:-4]+'seg.nii.gz'))
        if args.eval:
            vtk_files.append(os.path.join(vtk_folder, file_[:-4]+'graph.vtp'))

    config.DATA.MIXED = False
    dataset = build_vessel_data(config, mode='test', debug=False, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        collate_fn=lambda x:
        image_graph_collate(x, gaussian_augment=config.DATA.MIXED),
        pin_memory=True)

    if args.general_transformer:
        net = build_model(config, general_transformer=True, pretrain=False, output_dim=3).to(device)
    else:
        net = build_model(config).to(device)
    
    # print('Loading model from:', args.model)
    checkpoint = torch.load(args.model, map_location='cpu')
    net.load_state_dict(checkpoint['net'], strict=not args.no_strict_loading)
    net.eval()  # Put the CNN in evaluation mode
    
    t_start = time.time()
    sinkhorn_distance = SinkhornDistance(eps=1e-7, max_iter=100)
    
    metrics = tuple([COCOMetric(classes=['Node'], per_class=False, verbose=False)])
    iou_thresholds = get_unique_iou_thresholds(metrics)
    iou_mapping = get_indices_of_iou_for_each_metric(iou_thresholds, metrics)
    box_evaluator = box_ap(box_iou_np, iou_thresholds, max_detections=100)
    
    mean_smd = []
    node_ap_result = []
    edge_ap_result = []
    beta_errors = []

    folds_smd = []
    folds_node_mAP = []
    folds_node_mAR = []
    folds_edge_mAP = []
    folds_edge_mAR = []

    fold_size = int(len(dataset)/ 5)
    print(len(dataset))
    print(fold_size)
    for idx, batchdata in tqdm(enumerate(loader)):
        images, segs, nodes, edges, z_pos, domains = batchdata[0], batchdata[1], batchdata[2], batchdata[3], batchdata[4], batchdata[5]
        
        # # inputs, targets = self.get_batch(batchdata, image_keys=IMAGE_KEYS, label_keys="label")
        # # inputs = torch.cat(inputs, 1)
        images = images.to(device,  non_blocking=False)
        segs = segs.to(device,  non_blocking=False)
        nodes = [node.to(device,  non_blocking=False) for node in nodes]
        boxes = [torch.cat([node, 0.2*torch.ones(node.shape, device=node.device)], dim=-1) for node in nodes]
        edges = [edge.to(device,  non_blocking=False) for edge in edges]
        domains = domains.to(device, non_blocking=False)
        net.eval()

        if args.seg:
            h, out, _, _, _, _ = net(segs.type(torch.FloatTensor).to(device), z_pos, domain_labels=domains)
        else:
            h, out, _, _, _, _ = net(images.type(torch.FloatTensor).to(device), z_pos, domain_labels=domains)

        out = relation_infer(h.detach(), out, net, config.MODEL.DECODER.OBJ_TOKEN, config.MODEL.DECODER.RLN_TOKEN, apply_nms=False)

        if args.eval:
            for sample_num in range(images.shape[0]):
                pred_nodes = torch.tensor(out['pred_nodes'][sample_num], dtype=torch.float)
                pred_edges = torch.tensor(out['pred_rels'][sample_num], dtype=torch.int64)

                sample_nodes =nodes[sample_num]
                sample_edges = edges[sample_num]
                boxes = [torch.cat([sample_nodes, 0.2*torch.ones(sample_nodes.shape, device=sample_nodes.device)], dim=-1).cpu().numpy()]
                boxes_class = [np.zeros(boxes[0].shape[0])]
                edge_boxes = torch.stack([sample_nodes[sample_edges[:,0]], sample_nodes[sample_edges[:,1]]], dim=2)
                edge_boxes = torch.cat([torch.min(edge_boxes, dim=2)[0]-0.1, torch.max(edge_boxes, dim=2)[0]+0.1], dim=-1).cpu().numpy()
                edge_boxes = [edge_boxes[:,[0,1,3,4,2,5]]]

                if pred_edges.shape[0]>0:
                    pred_edge_boxes = torch.stack([pred_nodes[pred_edges[:,0]], pred_nodes[pred_edges[:,1]]], dim=2)
                    pred_edge_boxes = torch.cat([torch.min(pred_edge_boxes, dim=2)[0]-0.1, torch.max(pred_edge_boxes, dim=2)[0]+0.1], dim=-1).numpy()
                    pred_edge_boxes = [pred_edge_boxes[:,[0,1,3,4,2,5]]]
                    edge_boxes_class = [np.zeros(sample_edges.shape[0])]
                else:
                    pred_edge_boxes = []
                    edge_boxes_class = []
                # boxes_scores = [np.ones(boxes[0].shape[0])]

                # Calculate betti numbers
                # G1 = nx.Graph()
                # G1.add_nodes_from([i for i, n in enumerate(sample_nodes)])
                # G1.add_edges_from([tuple(e) for e in sample_edges.cpu().tolist()])
                # connected_components = len(list(nx.connected_components(G1)))
                # beta_gt = np.array([connected_components, len(G1.edges) + connected_components - len(G1.nodes)])

                # G2 = nx.Graph()
                # G2.add_nodes_from([i for i, n in enumerate(pred_nodes)])
                # G2.add_edges_from([(e[0].item(), e[1].item()) for e in pred_edges])
                # connected_components = len(list(nx.connected_components(G2)))
                # beta_pred = np.array([connected_components, len(G2.edges) + connected_components - len(G2.nodes)])
                # beta_errors.append(2 * np.abs(beta_pred - beta_gt) / (beta_gt + beta_pred + 1e-10))

                # mean AP
                node_ap = box_evaluator([out["pred_boxes"][sample_num]], [out["pred_boxes_class"][sample_num]], [out["pred_boxes_score"][sample_num]], boxes, boxes_class)
                node_ap_result.extend(node_ap)

                # mean AP
                edge_ap = box_evaluator(pred_edge_boxes, [out["pred_rels_class"][sample_num]], [out["pred_rels_score"][sample_num]], edge_boxes, edge_boxes_class, convert_box=False)
                edge_ap_result.extend(edge_ap)
                
                # mean SMD            
                A = torch.zeros((sample_nodes.shape[0], sample_nodes.shape[0]))
                pred_A = torch.zeros((pred_nodes.shape[0], pred_nodes.shape[0]))

                A[sample_edges[:,0],sample_edges[:,1]] = 1
                A[sample_edges[:,1],sample_edges[:,0]] = 1
                A = torch.tril(A)

                if sample_nodes.shape[0]>1 and pred_nodes.shape[0]>1 and pred_edges.size != 0:
                    # print(pred_edges)
                    pred_A[pred_edges[:,0], pred_edges[:,1]] = 1.0
                    pred_A[pred_edges[:,1], pred_edges[:,0]] = 1.0
                    pred_A = torch.tril(pred_A)

                    mean_smd.append(compute_meanSMD(A, sample_nodes, pred_A, pred_nodes, sinkhorn_distance, n_points=100).numpy())

                # Calculate mean metrics for each fold
                if (idx * config.DATA.BATCH_SIZE + sample_num) % fold_size == (fold_size - 1):
                    print("Calculating intermediate results")
                    # accumulate AP score
                    node_metric_scores = {}
                    edge_metric_scores = {}
                    for metric_idx, metric in enumerate(metrics):
                        _filter = partial(iou_filter, iou_idx=iou_mapping[metric_idx])
                        iou_filtered_results = list(map(_filter, node_ap_result))
                        score, curve = metric(iou_filtered_results)
                        if score is not None:
                            node_metric_scores.update(score)

                        iou_filtered_results = list(map(_filter, edge_ap_result))
                        score, curve = metric(iou_filtered_results)
                        if score is not None:
                            edge_metric_scores.update(score)
                    folds_node_mAP.append(node_metric_scores['mAP_IoU_0.50_0.95_0.05_MaxDet_40'][0])
                    folds_node_mAR.append(node_metric_scores['mAR_IoU_0.50_0.95_0.05_MaxDet_40'][0])
                    folds_edge_mAP.append(edge_metric_scores['mAP_IoU_0.50_0.95_0.05_MaxDet_40'][0])
                    folds_edge_mAR.append(edge_metric_scores['mAR_IoU_0.50_0.95_0.05_MaxDet_40'][0])

                    # Accumulate SMD score
                    smd_mean = torch.tensor(mean_smd).mean()

                    folds_smd.append(smd_mean)

                    # Reset lists
                    mean_smd = []
                    node_ap_result = []
                    edge_ap_result = []

    # print metrics
    smd = torch.tensor(folds_smd).mean()
    smd_std = torch.tensor(folds_smd).std()
    node_mAP = torch.tensor(folds_node_mAP).mean()
    node_mAP_std = torch.tensor(folds_node_mAP).std()
    node_mAR = torch.tensor(folds_node_mAR).mean()
    node_mAR_std = torch.tensor(folds_node_mAR).std()
    edge_mAP = torch.tensor(folds_edge_mAP).mean()
    edge_mAP_std = torch.tensor(folds_edge_mAP).std()
    edge_mAR = torch.tensor(folds_edge_mAR).mean()
    edge_mAR_std = torch.tensor(folds_edge_mAR).std()

    print("smd: ", torch.tensor(folds_smd).mean())
    print("smd std: ", torch.tensor(folds_smd).std())
    print("node mAP: ", torch.tensor(folds_node_mAP).mean())
    print("node mAP std: ", torch.tensor(folds_node_mAP).std())
    print("node mAR: ", torch.tensor(folds_node_mAR).mean())
    print("node mAR std: ", torch.tensor(folds_node_mAR).std())
    print("edge mAP: ", torch.tensor(folds_edge_mAP).mean())
    print("edge mAP std: ", torch.tensor(folds_edge_mAP).std())

    # print("node scores:")
    # print(json.dumps(node_metric_scores, sort_keys=True, indent=4))
    # print("####################################################################################")
    # print("Edge scores:")
    # print(json.dumps(edge_metric_scores, sort_keys=True, indent=4))

    # b0, b1 = np.mean(beta_errors, axis=0)
    # b0_std, b1_std = np.std(beta_errors, axis=0)

    # print("Betti-error:", b0, b1)
    # print("Betti-error std:", b0_std, b1_std)

    csv_value_string = f'{smd};{smd_std}'
    csv_header_string = f'smd;smd-(std)'

    csv_value_string += f';{node_mAP};{node_mAP_std};{node_mAR};{node_mAR_std};{edge_mAP};{edge_mAP_std};{edge_mAR};{edge_mAR_std}'
    csv_header_string += f';node_mAP;node_mAP_std;node_mAR;node_mAR_std;edge_mAP;edge_mAP_std;edge_mAR;edge_mAR_std'

    for fold_no in range(5):
        csv_value_string += f';{folds_smd[fold_no]};{folds_node_mAP[fold_no]};{folds_node_mAR[fold_no]};{folds_edge_mAP[fold_no]};{folds_edge_mAP[fold_no]}'
        csv_header_string += f';fold{fold_no}_smd;fold{fold_no}_node_mAP;fold{fold_no}_node_mAR;fold{fold_no}_edge_mAP;fold{fold_no}_edge_mAR'

    # for field in node_metric_scores:
    #     csv_header_string += f';node_{field};node_{field}-(std)'
    #     csv_value_string += f';{node_metric_scores[field][0]};{node_metric_scores[field][1]}'

    # for field in edge_metric_scores:
    #     csv_header_string += f';edge_{field};edge_{field}-(std)'
    #     csv_value_string += f';{edge_metric_scores[field][0]};{edge_metric_scores[field][1]}'

    print(csv_header_string)
    print(csv_value_string)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)