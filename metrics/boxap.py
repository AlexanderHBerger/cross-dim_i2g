
import torch
import math
from torch import nn
from pathlib import Path
from functools import partial
from typing import Optional, Sequence, Callable, Dict, List, Tuple
from monai.config import IgniteInfo
from monai.utils import min_version, optional_import
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
from monai.config import TensorOrList
from .box_ops_np import box_iou_np
from .matching import matching_batch
import numpy as np
import warnings
reinit__is_reduced, _ = optional_import(
    "ignite.metrics.metric", IgniteInfo.OPT_IMPORT_VERSION, min_version, "reinit__is_reduced"
)
if TYPE_CHECKING:
    from ignite.engine import Engine
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
import pdb
from abc import ABC

# @self._engine.on(Events.EPOCH_COMPLETED)

class MeanSingleAP(Metric):
    def __init__(self, computer, output_transform: Callable = lambda x: x, edge_ap=False, metric="mAP_IoU_0.50_0.95_0.05_MaxDet_40"):
        self.computer = computer
        self.edge_ap = edge_ap
        self.metric = metric
        super().__init__(output_transform=output_transform)

    def compute(self):
        if self.edge_ap:
            return self.computer.get_edge_ap()[self.metric]
        else:
            return self.computer.get_node_ap()[self.metric]

    def reset(self):
        return

    def update(self, output):
        return

from .coco import COCOMetric
class MeanBoxAP(Metric):
    """[summary]

    Args:
        Metric ([type]): [description]
    """
    def __init__(self,
                 on_edges: bool = False,
                 output_transform: Callable = lambda x: x,
                 max_detections: int = 100):
        """[summary]

        Args:
            output_transform (Callable, optional): [description]. Defaults to lambdax:x.
        """
        self.iou_fn = box_iou_np
        metrics = tuple([COCOMetric(classes=['Node'], per_class=False, verbose=False)])
        self.metric_fn_nodes = BoxAP(convert_box=True,
                               metrics=metrics,
                               reduction='none',
                               max_detections=max_detections
                              )
        self.metric_fn_edges = BoxAP(convert_box=False,
                                     metrics=metrics,
                                     reduction='none',
                                     max_detections=max_detections
                                     )
        self.result_nodes = None
        self.result_edges = None
        super().__init__(output_transform=output_transform,)

    @reinit__is_reduced
    def reset(self) -> None:
        self.metric_fn_nodes.reset()
        self.metric_fn_edges.reset()

    @reinit__is_reduced
    def update(self, output) -> None:
        """[summary]

        Args:
            output ([type]): [description]

        Returns:
            [type]: [description]
        """
        pred_boxes, \
        pred_classes, \
        pred_scores, \
        gt_boxes, \
        gt_classes, \
        pred_nodes_batch, \
        pred_edges_batch, \
        pred_edge_class_batch, \
        pred_edge_score_batch, \
        nodes_batch, \
        edges_batch = output

        pred_nodes_batch = [pred_nodes for pred_nodes in pred_nodes_batch]
        edge_boxes_batch = [torch.stack([nodes[edges_batch[i][:, 0]], nodes[edges_batch[i][:, 1]]], dim=2) for i, nodes in enumerate(nodes_batch)]
        edge_boxes_batch = [torch.cat([torch.min(edge_boxes, dim=2)[0] - 0.1, torch.max(edge_boxes, dim=2)[0] + 0.1],
                                      dim=-1) for edge_boxes in edge_boxes_batch]

        pred_edge_boxes_batch = [torch.stack([torch.from_numpy(pred_nodes[pred_edges_batch[i][:, 0]]), torch.from_numpy(pred_nodes[pred_edges_batch[i][:, 1]])], dim=2) for i, pred_nodes in enumerate(pred_nodes_batch)]
        pred_edge_boxes_batch = [torch.cat(
            [torch.min(pred_edge_boxes, dim=2)[0] - 0.1, torch.max(pred_edge_boxes, dim=2)[0] + 0.1],
            dim=-1).cpu().numpy() for pred_edge_boxes in pred_edge_boxes_batch]
        if edge_boxes_batch[0].shape[1] == 6:
            edge_boxes_batch = [torch.stack([edge_boxes[:, [0, 1, 3, 4, 2, 5]]])[0].cpu() for edge_boxes in
                                edge_boxes_batch]
            pred_edge_boxes_batch = [np.array([pred_edge_boxes[:, [0, 1, 3, 4, 2, 5]]])[0] for pred_edge_boxes in
                                     pred_edge_boxes_batch]
        elif edge_boxes_batch[0].shape[1] == 4:
            edge_boxes_batch = [torch.stack([edge_boxes[:, [0, 1, 2, 3]]])[0].cpu() for edge_boxes in
                                edge_boxes_batch]
            pred_edge_boxes_batch = [np.array([pred_edge_boxes[:, [0, 1, 2, 3]]])[0] for pred_edge_boxes in
                                     pred_edge_boxes_batch]
        else:
            print("ERROR: Unknown shape")

        edge_boxes_class_batch = [np.array([np.zeros(edges.shape[0])])[0] for edges in edges_batch]
        self.metric_fn_edges(
            pred_edge_boxes_batch,
            [np.array(pred_edge_class) for pred_edge_class in pred_edge_class_batch],
            pred_edge_score_batch,
            edge_boxes_batch,
            edge_boxes_class_batch
        )

        self.metric_fn_nodes(pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes)

    def get_node_ap(self):
        return self.result_nodes

    def get_edge_ap(self):
        return self.result_edges

    # TODO: change to adapt multiple metrics at the same time
    def compute(self) -> Any:
        """[summary]

        Raises:
            RuntimeError: [description]

        Returns:
            Any: [description]
        """
        self.result_nodes = self.metric_fn_nodes.aggregate()
        self.result_edges = self.metric_fn_edges.aggregate()
        if isinstance(self.result_nodes, (tuple, list)):
            if len(self.result_nodes) > 1:
                warnings.warn("metric handler can only record the first value of result list.")
            self.result_nodes = self.result_nodes[0]
        if isinstance(self.result_edges, (tuple, list)):
            if len(self.result_edges) > 1:
                warnings.warn("metric handler can only record the first value of result list.")
            self.result_edges = self.result_edges[0]

        self._is_reduced = True

        # save score of every image into engine.state for other components
        """if False and self.save_details:
            if self._engine is None or self._name is None:
                raise RuntimeError("please call the attach() function to connect expected engine first.")
            for key in result.keys():
                print(key, result[key])
                self._engine.state.metric_details[key] = result[key]"""
        # TODO: currently returning only one result

        res = tuple([x / 2 for x in (self.result_nodes['mAP_IoU_0.50_0.95_0.05_MaxDet_40'] + self.result_edges['mAP_IoU_0.50_0.95_0.05_MaxDet_40'])])
        return res

    def attach(self, engine: Engine, name: str) -> None:
        """[summary]

        Args:
            engine (Engine): [description]
            name (str): [description]
        """        
        super().attach(engine=engine, name=name)
        # FIXME: record engine for communication, ignite will support it in the future version soon
        self._engine = engine
        self._name = name # TODO: incorporate all metric name
        if self.save_details and not hasattr(engine.state, "metric_details"):
            engine.state.metric_details = {}


class BoxAP(ABC):
    r"""[summary]

    Args:
        ABC ([type]): [description]

    Raises:
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """    
    def __init__(self,
                 convert_box: bool,
                 metrics: Sequence[ABC],
                 iou_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = box_iou_np,
                 reduction="mean",
                 max_detections=100
                 ):
        """[summary]

        Args:
            eps ([type]): [description]
            max_iter ([type]): [description]
            reduction (str, optional): [description]. Defaults to "mean".
        """
        super(BoxAP, self).__init__()
        self.reduction = reduction
        self.metrics = metrics
        self.iou_fn = iou_fn
        self.iou_thresholds = get_unique_iou_thresholds(metrics)
        self.iou_mapping = get_indices_of_iou_for_each_metric(self.iou_thresholds, metrics)
        self.box_ap = box_ap(iou_fn, self.iou_thresholds, reduction=reduction, max_detections=max_detections)
        self.convert_box = convert_box

        # the default one
        self.buffer_num: int = 0
        self._buffers = []  # store results of each image
        self._synced: bool = False

    def __call__(self, pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes):
        """[summary]

        Args:
            node_list ([type]): [description]
            edge_list ([type]): [description]
            pred_node_list ([type]): [description]
            pred_edge_list ([type]): [description]

        Returns:
            [type]: [description]
        """        

        ret = self.box_ap(pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes, convert_box=self.convert_box)
        self.add(ret)
        return ret

    def reset(self):
        """
        Reset the buffers for cumulative tensors and the synced results.

        """
        self._buffers = []
        self._synced = False

    def add(self, data: torch.Tensor):
        """
        Add samples to the cumulative buffers.

        Args:
            data: list of input tensor, make sure the input data order is always the same in a round.
                every item of data will be added to the corresponding buffer.

        """
        self._buffers.extend(data)
        
        self._synced = False

    def aggregate(self) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Execute reduction logic for the output of `compute_meandice`.

        """
        metric_scores = {}
        metric_curves = {}
        for metric_idx, metric in enumerate(self.metrics):
            _filter = partial(iou_filter, iou_idx=self.iou_mapping[metric_idx])
            iou_filtered_results = list(map(_filter, self._buffers))
            score, curve = metric(iou_filtered_results)
            if score is not None:
                metric_scores.update(score)
            
            if curve is not None:
                metric_curves.update(curve)
        return metric_scores #, metric_curves # TODO: need to check what to do with the curves

def get_unique_iou_thresholds(metrics):
    """
    Compute unique set of iou thresholds
    """
    iou_thresholds = [_i for i in metrics for _i in i.get_iou_thresholds()]
    iou_thresholds = list(set(iou_thresholds))
    iou_thresholds.sort()
    return iou_thresholds

def get_indices_of_iou_for_each_metric(iou_thresholds, metrics):
    """
    Find indices of iou thresholds for each metric
    """
    return [[iou_thresholds.index(th) for th in m.get_iou_thresholds()]
            for m in metrics]

def iou_filter(image_dict: Dict[int, Dict[str, np.ndarray]], iou_idx: List[int],
                filter_keys: Sequence[str] = ('dtMatches', 'gtMatches', 'dtIgnore')):
    """
    This functions can be used to filter specific IoU values from the results
    to make sure that the correct IoUs are passed to metric
    
    Parameters
    ----------
    image_dict : dict
        dictionary containin :param:`filter_keys` which contains IoUs in the first dimension
    iou_idx : List[int]
        indices of IoU values to filter from keys
    filter_keys : tuple, optional
        keys to filter, by default ('dtMatches', 'gtMatches', 'dtIgnore')
    
    Returns
    -------
    dict
        filtered dictionary
    """
    iou_idx = list(iou_idx)
    filtered = {}
    for cls_key, cls_item in image_dict.items():
        filtered[cls_key] = {key: item[iou_idx] if key in filter_keys else item
                                for key, item in cls_item.items()}
    return filtered


class box_ap(nn.Module):
    def __init__(self, 
                iou_fn,
                iou_thresholds,
                reduction='none',
                max_detections=100):
        super(box_ap, self).__init__()
        self.iou_fn = iou_fn
        self.iou_thresholds = iou_thresholds
        self.max_detections = max_detections
        self.reduction = reduction

    def forward(self,
                pred_boxes: Sequence[np.ndarray],
                pred_classes: Sequence[np.ndarray],
                pred_scores: Sequence[np.ndarray],
                gt_boxes: Sequence[np.ndarray],
                gt_classes: Sequence[np.ndarray],
                gt_ignore: Sequence[Sequence[bool]] = None,
                convert_box: bool=True,
                ):
                
        if gt_ignore is None:
            n = [0 if gt_boxes_img.size == 0 else gt_boxes_img.shape[0] for gt_boxes_img in gt_boxes]
            gt_ignore = [np.zeros(_n).reshape(-1) for _n in n]
        return matching_batch(
            self.iou_fn, self.iou_thresholds, pred_boxes=pred_boxes, pred_classes=pred_classes,
            pred_scores=pred_scores, gt_boxes=gt_boxes, gt_classes=gt_classes, gt_ignore=gt_ignore,
            max_detections=self.max_detections, convert_box=convert_box)
