import os
import yaml
import json
from argparse import ArgumentParser
import pdb
import numpy as np
import networkx as nx
from data.dataset_real_eye_vessels import build_real_vessel_network_data

parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config. See /config for examples.')
parser.add_argument('--checkpoint', default=None, help='checkpoint of the model to test.')
parser.add_argument('--device', default='cuda',
                        help='device to use for training')
parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0,1],
                        help='list of index where skip conn will be made.')
parser.add_argument('--save_path', default=None, help='checkpoint of the model to test.')
parser.add_argument("--max_samples", default=0, help='On how many samples should the net be trained?', type=int)
parser.add_argument('--pretrain_seg', default=False, action="store_true",
                    help="Whether to pretrain on segs instead of raw images")
parser.add_argument('--no_strict_loading', default=False, action="store_true",
                    help="Whether the model was pretrained with domain adversarial. If true, the checkpoint will be loaded with strict=false")



class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def ensure_format(bboxes):
    boxes_new = []
    for bbox in bboxes:
        if bbox[0] > bbox[2]:
            bbox[0], bbox[2] = bbox[2], bbox[0]
        if bbox[1] > bbox[3]:
            bbox[1], bbox[3] = bbox[3], bbox[1]
        
        # to take care of horizontal and vertical edges
        if bbox[2]-bbox[0]<0.2:
            bbox[0] = bbox[0]-0.075
            bbox[2] = bbox[2]+0.075
        if bbox[3]-bbox[1]<0.2:
            bbox[1] = bbox[1]-0.075
            bbox[3] = bbox[3]+0.075
            
        boxes_new.append(bbox)
    return np.array(boxes_new)

def test(args):
    
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['message'])
    config = dict2obj(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))

    import torch
    from monai.data import DataLoader
    from tqdm import tqdm
    import numpy as np

    from data.dataset_road_network import build_road_network_data
    from data.dataset_synthetic_eye_vessels import build_synthetic_vessel_network_data
    from models import build_model
    from training.inference import relation_infer
    from metrics.metric_smd import StreetMoverDistance
    from metrics.metric_map import BBoxEvaluator
    from utils.box_ops_2D import box_cxcywh_to_xyxy_np
    from utils.utils import image_graph_collate_road_network
    from metrics.topo import compute_topo

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda") if args.device=='cuda' else torch.device("cpu")

    config.DATA.MIXED = False

    net = build_model(config).to(device)

    config.DATA.MIXED = False

    if config.DATA.DATASET == 'road_dataset' or config.DATA.DATASET == 'mixed_road_dataset':
        build_dataset_function = build_road_network_data
    elif config.DATA.DATASET == 'synthetic_eye_vessel_dataset' or config.DATA.DATASET == 'mixed_synthetic_eye_vessel_dataset':
        build_dataset_function = build_synthetic_vessel_network_data
    elif config.DATA.DATASET == 'real_eye_vessel_dataset' or config.DATA.DATASET == 'mixed_real_eye_vessel_dataset':
        build_dataset_function = build_real_vessel_network_data

    test_ds = build_dataset_function(
        config, mode='test', max_samples=args.max_samples, use_grayscale=args.pretrain_seg
    )

    # The batch size is 1 such that we can leave the same implementation for the folding standard deviation
    batch_size = 1
    test_loader = DataLoader(test_ds,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=config.DATA.NUM_WORKERS,
                            collate_fn=image_graph_collate_road_network,
                            pin_memory=True)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(checkpoint['net'], strict=not args.no_strict_loading)
    net.eval()

    # init metric
    # metric = StreetMoverDistance(eps=1e-7, max_iter=100, reduction=MetricReduction.MEAN)
    metric_smd = StreetMoverDistance(eps=1e-5, max_iter=10, reduction='none')
    smd_results = []

    metric_node_map = BBoxEvaluator(['node'], max_detections=100)
    metric_edge_map = BBoxEvaluator(['edge'], max_detections=100)

    topo_results = []
    beta_errors = []

    folds_smd = []
    folds_node_mAP = []
    folds_node_mAR = []
    folds_edge_mAP = []
    folds_edge_mAR = []
    folds_topo_precision = []
    folds_topo_recall = []

    fold_size = int(len(test_ds)/ 5)
    print(len(test_ds))
    print(fold_size)


    with torch.no_grad():
        print('Started processing test set.')
        for i, batchdata in enumerate(tqdm(test_loader)):

            # extract data and put to device
            images, segs, nodes, edges = batchdata[0], batchdata[1], batchdata[2], batchdata[3]
            images = images.to(args.device,  non_blocking=False)
            segs = segs.to(args.device,  non_blocking=False)
            nodes = [node.to(args.device,  non_blocking=False) for node in nodes]
            edges = [edge.to(args.device,  non_blocking=False) for edge in edges]

            h, out, _, _, _, _ = net(images, seg=False)
            pred_nodes, pred_edges, pred_nodes_box, pred_nodes_box_score, pred_nodes_box_class, pred_edges_box_score, pred_edges_box_class = relation_infer(
                h.detach(), out, net, config.MODEL.DECODER.OBJ_TOKEN, config.MODEL.DECODER.RLN_TOKEN,
                nms=False, map_=True
            )

            # Add smd of current batch elem
            ret = metric_smd(nodes, edges, pred_nodes, pred_edges)
            smd_results += ret.tolist()

            # Add elements of current batch elem to node map evaluator
            metric_node_map.add(
                pred_boxes=[box_cxcywh_to_xyxy_np(np.concatenate([nodes_.cpu().numpy(), np.ones_like(nodes_.cpu()) * 0.2], axis=1)) for nodes_ in pred_nodes],
                pred_classes=pred_nodes_box_class,
                pred_scores=pred_nodes_box_score,
                gt_boxes=[box_cxcywh_to_xyxy_np(np.concatenate([nodes_.cpu().numpy(), np.ones_like(nodes_.cpu()) * 0.2], axis=1)) for nodes_ in nodes],
                gt_classes=[np.ones((nodes_.shape[0],)) for nodes_ in nodes]
            )

            # Calculate betti scores
            # for nodes_sample, edges_sample, pred_nodes_sample, pred_edges_sample in zip(nodes, edges, pred_nodes, pred_edges):
            #     G1 = nx.Graph()
            #     G1.add_nodes_from([i for i, n in enumerate(nodes_sample)])
            #     G1.add_edges_from([tuple(e) for e in edges_sample.cpu().tolist()])
            #     connected_components = len(list(nx.connected_components(G1)))
            #     beta_gt = np.array([connected_components, len(G1.edges) + connected_components - len(G1.nodes)])

            #     G2 = nx.Graph()
            #     G2.add_nodes_from([i for i, n in enumerate(pred_nodes_sample)])
            #     G2.add_edges_from([tuple(e) for e in pred_edges_sample])
            #     connected_components = len(list(nx.connected_components(G2)))
            #     beta_pred = np.array([connected_components, len(G2.edges) + connected_components - len(G2.nodes)])
            #     beta_errors.append(2 * np.abs(beta_pred - beta_gt) / (beta_gt + beta_pred + 1e-10))

            # Add elements of current batch elem to edge map evaluator
            pred_edges_box = []
            for edges_, nodes_ in zip(pred_edges, pred_nodes):
                nodes_ = nodes_.cpu().numpy()
                edges_box = ensure_format(np.hstack([nodes_[edges_[:, 0]], nodes_[edges_[:, 1]]]))
                pred_edges_box.append(edges_box)

            gt_edges_box = []
            for edges_, nodes_ in zip(edges, nodes):
                nodes_ , edges_ = nodes_.cpu().numpy(), edges_.cpu().numpy()
                edges_box = ensure_format(np.hstack([nodes_[edges_[:, 0]], nodes_[edges_[:, 1]]]))
                gt_edges_box.append(edges_box)

            metric_edge_map.add(
                pred_boxes=pred_edges_box,
                pred_classes=pred_edges_box_class,
                pred_scores=pred_edges_box_score,
                gt_boxes=gt_edges_box,
                gt_classes=[np.ones((edges_.shape[0],)) for edges_ in edges]
            )
        
            for node_, edge_, pred_node_, pred_edge_ in zip(nodes, edges, pred_nodes, pred_edges):
                topo_results.append(compute_topo(node_.cpu(), edge_.cpu(), pred_node_, pred_edge_))

            if i % fold_size == (fold_size - 1):
                print('Fold finished')
                topo_array=np.array(topo_results)
                folds_topo_precision.append(topo_array.mean(0)[0])
                folds_topo_recall.append(topo_array.mean(0)[1])

                folds_smd.append(torch.tensor(smd_results).mean().item())

                # Determine node box ap / ar
                node_metric_scores = metric_node_map.eval()
                edge_metric_scores = metric_edge_map.eval()

                folds_node_mAP.append(node_metric_scores['mAP_IoU_0.50_0.95_0.05_MaxDet_100'][0])
                folds_node_mAR.append(node_metric_scores['mAR_IoU_0.50_0.95_0.05_MaxDet_100'][0])
                folds_edge_mAP.append(edge_metric_scores['mAP_IoU_0.50_0.95_0.05_MaxDet_100'][0])
                folds_edge_mAR.append(edge_metric_scores['mAR_IoU_0.50_0.95_0.05_MaxDet_100'][0])

                # Reset metrics
                smd_results = []
                topo_results = []
                metric_node_map.reset()
                metric_edge_map.reset()



    print('Finished processing test set.')
    print(len(folds_smd))
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
    topo_precision = torch.tensor(folds_topo_precision).mean()
    topo_precision_std = torch.tensor(folds_topo_precision).std()
    topo_recall = torch.tensor(folds_topo_recall).mean()
    topo_recall_std = torch.tensor(folds_topo_recall).std()

    print("smd: ", torch.tensor(folds_smd).mean().item())
    print("smd std: ", torch.tensor(folds_smd).std().item())
    print("node mAP: ", torch.tensor(folds_node_mAP).mean().item())
    print("node mAP std: ", torch.tensor(folds_node_mAP).std().item())
    print("node mAR: ", torch.tensor(folds_node_mAR).mean().item())
    print("node mAR std: ", torch.tensor(folds_node_mAR).std().item())
    print("edge mAP: ", torch.tensor(folds_edge_mAP).mean().item())
    print("edge mAP std: ", torch.tensor(folds_edge_mAP).std().item())
    print("edge mAR: ", torch.tensor(folds_edge_mAR).mean().item())
    print("edge mAR std: ", torch.tensor(folds_edge_mAR).std().item())
    print("topo precision: ", torch.tensor(folds_topo_precision).mean().item())
    print("topo precision std: ", torch.tensor(folds_topo_precision).std().item())
    print("topo recall: ", torch.tensor(folds_topo_recall).mean().item())
    print("topo recall std: ", torch.tensor(folds_topo_recall).std().item())

    csv_value_string = f'{smd};{smd_std};{topo_precision};{topo_precision_std};{topo_recall};{topo_recall_std};{node_mAP};{node_mAP_std};{node_mAR};{node_mAR_std};{edge_mAP};{edge_mAP_std};{edge_mAR};{edge_mAR_std}'
    csv_header_string = f'smd;smd-(std);topo-prec;topo-prec-(std);topo-rec;topo-rec-(std);node-mAP;node-mAP-(std);node-mAR;node-mAR-(std);edge-mAP;edge-mAP-(std);edge-mAR;edge-mAR-(std)'

    for fold_no in range(5):
        csv_value_string += f';{folds_smd[fold_no]};{folds_topo_precision[fold_no]};{folds_topo_recall[fold_no]};{folds_node_mAP[fold_no]};{folds_node_mAR[fold_no]};{folds_edge_mAP[fold_no]};{folds_edge_mAP[fold_no]}'
        csv_header_string += f';fold{fold_no}_smd;fold{fold_no}_topo_precision;fold{fold_no}_topo_recall;fold{fold_no}_node_mAP;fold{fold_no}_node_mAR;fold{fold_no}_edge_mAP;fold{fold_no}_edge_mAR'


    # topo_array=np.array(topo_results)
    # print(f'topo mean: {topo_array.mean(0)}')
    # print(f'topo std: {topo_array.std(0)}')
    # # Determine smd
    # smd_mean = torch.tensor(smd_results).mean().item()
    # smd_std = torch.tensor(smd_results).std().item()
    # print(f'smd value: mean {smd_mean}, std {smd_std}\n')

    # # Determine node box ap / ar
    # node_metric_scores = metric_node_map.eval()
    # edge_metric_scores = metric_edge_map.eval()


    # print("Node scores")
    # print(json.dumps(node_metric_scores, sort_keys=True, indent=4))
    # print("####################################################################################")
    # print("Edge scores")
    # print(json.dumps(edge_metric_scores, sort_keys=True, indent=4))

    # b0, b1 = np.mean(beta_errors, axis=0)
    # b0_std, b1_std = np.std(beta_errors, axis=0)

    # print("Betti-error:", b0, b1)
    # print("Betti-error std:", b0_std, b1_std)

    # csv_value_string = f'{smd_mean};{smd_std}'
    # csv_header_string = f'smd;smd-(std)'

    # csv_value_string += f';{topo_array.mean(0)[0]};{topo_array.std(0)[0]};{topo_array.mean(0)[1]};{topo_array.std(0)[1]}'
    # csv_header_string += f';topo-prec;topo-pred-(std);topo-rec;topo-rec-(std)'

    # csv_value_string += f';{b0};{b0_std};{b1};{b1_std}'
    # csv_header_string += f';b0;b0-(std);b1;b1-(std)'

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
    test(args)
