"""
Some parts are adapted from https://github.com/cocodataset/cocoapi :

Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met: 

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies, 
either expressed or implied, of the FreeBSD Project.
"""
"""
For the remaining parts:

Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import time
import numpy as np
from loguru import logger
from typing import Sequence, List, Dict, Union, Tuple
import numpy as np
from abc import ABC
import pdb

class COCOMetric(ABC):
    def __init__(self,
                 classes: Sequence[str],
                 iou_list: Sequence[float] = (0.05, 1.0, 0.05),
                 iou_ranges: Sequence[Sequence[float]] = [(0.05, 0.5, 0.05), (0.5, 0.95, 0.05)],
                 max_detection: Sequence[int] = (40,),
                 per_class: bool = True,
                 verbose: bool = True):
        """
        Class to compute COCO metrics
        Metrics computed:
            mAP over the IoU range specified by :param:`iou_range` at last value of :param:`max_detection`
            AP values at IoU thresholds specified by :param:`iou_list` at last value of :param:`max_detection`
            AR over max detections thresholds defined by :param:`max_detection` (over iou range)

        Args:
            classes (Sequence[str]): name of each class (index needs to correspond to predicted class indices!)
            iou_list (Sequence[float]): specific thresholds where ap is evaluated and saved
            iou_range (Sequence[float]): (start, stop, step) for mAP iou thresholds
            max_detection (Sequence[int]): maximum number of detections per image
            verbose (bool): log time needed for evaluation
        """
        self.verbose = verbose
        self.classes = classes
        self.per_class = per_class

        iou_list = np.arange(*iou_list)
        self.iou_thresholds = iou_list
        for iou_range in iou_ranges:
            _iou_range = np.linspace(iou_range[0], iou_range[1],
                                     int(np.round((iou_range[1] - iou_range[0]) / iou_range[2])) + 1, endpoint=True)
            self.iou_thresholds = np.union1d(self.iou_thresholds, _iou_range)
        self.iou_ranges = iou_ranges

        # get indices of iou values of ious range and ious list for later evaluation
        self.iou_list_idx = np.nonzero(iou_list[:, np.newaxis] == self.iou_thresholds[np.newaxis])[1]

        self.iou_range_idxs = []
        for iou_range in iou_ranges:
            _iou_range = np.linspace(iou_range[0], iou_range[1],
                                     int(np.round((iou_range[1] - iou_range[0]) / iou_range[2])) + 1, endpoint=True)
            self.iou_range_idxs.append(np.nonzero(_iou_range[:, np.newaxis] == self.iou_thresholds[np.newaxis])[1])

        assert (self.iou_thresholds[self.iou_list_idx] == iou_list).all()

        self.recall_thresholds = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.max_detections = max_detection

    def __call__(self, *args, **kwargs) -> (Dict[str, float], Dict[str, np.ndarray]):
        """
        Compute metric. See :func:`compute` for more information.
        Args:
            *args: positional arguments passed to :func:`compute`
            **kwargs: keyword arguments passed to :func:`compute`
        Returns:
            Dict[str, float]: dictionary with scalar values for evaluation
            Dict[str, np.ndarray]: dictionary with arrays, e.g. for visualization of graphs
        """
        return self.compute(*args, **kwargs)

    def check_number_of_iou(self, *args) -> None:
        """
        Check if shape of input in first dimension is consistent with expected IoU values
        (assumes IoU dimension is the first dimension)

        Args:
            args: array like inputs with shape function
        """
        num_ious = len(self.get_iou_thresholds())
        for arg in args:
            assert arg.shape[0] == num_ious

    def get_iou_thresholds(self) -> Sequence[float]:
        """
        Return IoU thresholds needed for this metric in an numpy array

        Returns:
            Sequence[float]: IoU thresholds [M], M is the number of thresholds
        """
        return self.iou_thresholds

    def compute(self,
                results_list: List[Dict[int, Dict[str, np.ndarray]]],
                ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Compute COCO metrics

        Args:
            results_list (List[Dict[int, Dict[str, np.ndarray]]]): list with result s per image (in list)
                per category (dict). Inner Dict contains multiple results obtained by :func:`box_matching_batch`.
                `dtMatches`: matched detections [T, D], where T = number of thresholds, D = number of detections
                `gtMatches`: matched ground truth boxes [T, G], where T = number of thresholds, G = number of 
                    ground truth
                `dtScores`: prediction scores [D] detection scores
                `gtIgnore`: ground truth boxes which should be ignored [G] indicate whether ground truth
                    should be ignored
                `dtIgnore`: detections which should be ignored [T, D], indicate which detections should be ignored

        Returns:
            Dict[str, float]: dictionary with coco metrics
            Dict[str, np.ndarray]: None
        """
        if self.verbose:
            logger.info('Start COCO metric computation...')
            tic = time.time()

        dataset_statistics = self.compute_statistics(results_list=results_list)
        if self.verbose:
            toc = time.time()
            logger.info(f'Statistics for COCO metrics finished (t={(toc - tic):0.2f}s).')

        results = {}
        results.update(self.compute_ap(dataset_statistics))
        results.update(self.compute_ar(dataset_statistics))

        if self.verbose:
            toc = time.time()
            logger.info(f'COCO metrics computed in t={(toc - tic):0.2f}s.')
        return results, None

    def compute_ap(self, dataset_statistics: dict) -> dict:
        """
        Compute AP metrics

        Args:
            results_list (List[Dict[int, Dict[str, np.ndarray]]]): list with result s per image (in list)
                per category (dict). Inner Dict contains multiple results obtained by :func:`box_matching_batch`.
                `dtMatches`: matched detections [T, D], where T = number of thresholds, D = number of detections
                `gtMatches`: matched ground truth boxes [T, G], where T = number of thresholds, G = number of
                    ground truth
                `dtScores`: prediction scores [D] detection scores
                `gtIgnore`: ground truth boxes which should be ignored [G] indicate whether ground truth
                    should be ignored
                `dtIgnore`: detections which should be ignored [T, D], indicate which detections should be ignored
        """
        results = {}
        for i, iou_range in enumerate(self.iou_ranges): # mAP
            key = (f"mAP_IoU_{iou_range[0]:.2f}_{iou_range[1]:.2f}_{iou_range[2]:.2f}_"
                   f"MaxDet_{self.max_detections[-1]}")
            results[key] = self.select_ap(dataset_statistics, iou_idx=self.iou_range_idxs[i], max_det_idx=-1)

            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):  # per class results
                    key = (f"{cls_str}_"
                           f"mAP_IoU_{iou_range[0]:.2f}_{iou_range[1]:.2f}_{iou_range[2]:.2f}_"
                           f"MaxDet_{self.max_detections[-1]}")
                    results[key] = self.select_ap(dataset_statistics, iou_idx=self.iou_range_idxs[i],
                                                  cls_idx=cls_idx, max_det_idx=-1)

        for idx in self.iou_list_idx:   # AP@IoU
            key = f"AP_IoU_{self.iou_thresholds[idx]:.2f}_MaxDet_{self.max_detections[-1]}"
            results[key] = self.select_ap(dataset_statistics, iou_idx=[idx], max_det_idx=-1)

            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):  # per class results
                    key = (f"{cls_str}_"
                           f"AP_IoU_{self.iou_thresholds[idx]:.2f}_"
                           f"MaxDet_{self.max_detections[-1]}")
                    results[key] = self.select_ap(dataset_statistics,
                                                  iou_idx=[idx], cls_idx=cls_idx, max_det_idx=-1)
        return results

    def compute_ar(self, dataset_statistics: dict) -> dict:
        """
        Compute AR metrics

        Args:
            results_list (List[Dict[int, Dict[str, np.ndarray]]]): list with result s per image (in list)
                per category (dict). Inner Dict contains multiple results obtained by :func:`box_matching_batch`.
                `dtMatches`: matched detections [T, D], where T = number of thresholds, D = number of detections
                `gtMatches`: matched ground truth boxes [T, G], where T = number of thresholds, G = number of
                    ground truth
                `dtScores`: prediction scores [D] detection scores
                `gtIgnore`: ground truth boxes which should be ignored [G] indicate whether ground truth
                    should be ignored
                `dtIgnore`: detections which should be ignored [T, D], indicate which detections should be ignored
        """
        results = {}
        for max_det_idx, max_det in enumerate(self.max_detections):  # mAR
            for i, iou_range in enumerate(self.iou_ranges):
                key = f"mAR_IoU_{iou_range[0]:.2f}_{iou_range[1]:.2f}_{iou_range[2]:.2f}_MaxDet_{max_det}"
                results[key] = self.select_ar(dataset_statistics, max_det_idx=max_det_idx, iou_idx=self.iou_range_idxs[i])

                if self.per_class:
                    for cls_idx, cls_str in enumerate(self.classes):  # per class results
                        key = (f"{cls_str}_"
                               f"mAR_IoU_{iou_range[0]:.2f}_{iou_range[1]:.2f}_{iou_range[2]:.2f}_"
                               f"MaxDet_{max_det}")
                        results[key] = self.select_ar(dataset_statistics,
                                                      cls_idx=cls_idx, max_det_idx=max_det_idx, iou_idx=self.iou_range_idxs[i])

        for idx in self.iou_list_idx:   # AR@IoU
            key = f"AR_IoU_{self.iou_thresholds[idx]:.2f}_MaxDet_{self.max_detections[-1]}"
            results[key] = self.select_ar(dataset_statistics, iou_idx=idx, max_det_idx=-1)

            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):  # per class results
                    key = (f"{cls_str}_"
                           f"AR_IoU_{self.iou_thresholds[idx]:.2f}_"
                           f"MaxDet_{self.max_detections[-1]}")
                    results[key] = self.select_ar(dataset_statistics, iou_idx=idx,
                                                  cls_idx=cls_idx, max_det_idx=-1)
        return results

    @staticmethod
    def select_ap(dataset_statistics: dict, iou_idx: Union[int, List[int]] = None,
                  cls_idx: Union[int, Sequence[int]] = None, max_det_idx: int = -1) -> np.ndarray:
        """
        Compute average precision

        Args:
            dataset_statistics (dict): computed statistics over dataset
                `counts`: Number of thresholds, Number recall thresholds, Number of classes, Number of max
                    detection thresholds
                `recall`: Computed recall values [num_iou_th, num_classes, num_max_detections]
                `precision`: Precision values at specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes, num_max_detections]
                `scores`: Scores corresponding to specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes, num_max_detections]
            iou_idx: index of IoU values to select for evaluation(if None, all values are used)
            cls_idx: class indices to select, if None all classes will be selected
            max_det_idx (int): index to select max detection threshold from data

        Returns:
            np.ndarray: AP value
        """
        prec = dataset_statistics["precision"]
        if iou_idx is not None:
            prec = prec[:, iou_idx]
        if cls_idx is not None:
            prec = prec[..., cls_idx, :]
        prec = prec[..., max_det_idx]
        prec = np.mean(prec, axis=tuple(range(1, len(prec.shape))))
        return np.mean(prec), np.std(prec)

    @staticmethod
    def select_ar(dataset_statistics: dict, iou_idx: Union[int, Sequence[int]] = None,
                  cls_idx: Union[int, Sequence[int]] = None,
                  max_det_idx: int = -1) -> np.ndarray:
        """
        Compute average recall

        Args:
            dataset_statistics (dict): computed statistics over dataset
                `counts`: Number of thresholds, Number recall thresholds, Number of classes, Number of max
                    detection thresholds
                `recall`: Computed recall values [num_iou_th, num_classes, num_max_detections]
                `precision`: Precision values at specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes, num_max_detections]
                `scores`: Scores corresponding to specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes, num_max_detections]
            iou_idx: index of IoU values to select for evaluation(if None, all values are used)
            cls_idx: class indices to select, if None all classes will be selected
            max_det_idx (int): index to select max detection threshold from data

        Returns:
            np.ndarray: recall value
        """
        rec = dataset_statistics["recall"]
        if iou_idx is not None:
            rec = rec[:, iou_idx]
        if cls_idx is not None:
            rec = rec[..., cls_idx, :]
        rec = rec[..., max_det_idx]

        if len(rec[rec > -1]) == 0:
            rec = -1
            std = -1
        else:
            recs = np.zeros(rec.shape[0])
            for i, single_rec in enumerate(rec):
                recs[i] = np.mean(single_rec[single_rec > -1])
            rec = np.mean(recs)
            std = np.std(recs)

        return rec, std

    def compute_statistics(self, results_list: List[Dict[int, Dict[str, np.ndarray]]]
                           ) -> Dict[str, Union[np.ndarray, List]]:
        """
        Compute statistics needed for COCO metrics (mAP, AP of individual classes, mAP@IoU_Thresholds, AR)
        Adapted from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

        Args:
            results_list (List[Dict[int, Dict[str, np.ndarray]]]): list with result s per image (in list) 
                per cateory (dict). Inner Dict contains multiple results obtained by :func:`box_matching_batch`.
                `dtMatches`: matched detections [T, D], where T = number of thresholds, D = number of detections
                `gtMatches`: matched ground truth boxes [T, G], where T = number of thresholds, G = number of
                    ground truth
                `dtScores`: prediction scores [D] detection scores
                `gtIgnore`: ground truth boxes which should be ignored [G] indicate whether ground truth should be 
                    ignored
                `dtIgnore`: detections which should be ignored [T, D], indicate which detections should be ignored

        Returns:
            dict: computed statistics over dataset
                `counts`: Number of thresholds, Number recall thresholds, Number of classes, Number of max
                    detection thresholds
                `recall`: Computed recall values [num_iou_th, num_classes, num_max_detections]
                `precision`: Precision values at specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes, num_max_detections]
                `scores`: Scores corresponding to specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes, num_max_detections]
        """
        num_iou_th = len(self.iou_thresholds)
        num_recall_th = len(self.recall_thresholds)
        num_classes = len(self.classes)
        num_max_detections = len(self.max_detections)
        num_samples = len(results_list)

        # -1 for the precision of absent categories
        precision = -np.ones((num_samples, num_iou_th, num_recall_th, num_classes, num_max_detections))
        recall = -np.ones((num_samples, num_iou_th, num_classes, num_max_detections))
        scores = -np.ones((num_samples, num_iou_th, num_recall_th, num_classes, num_max_detections))
        # pdb.set_trace()
        for cls_idx, cls_i in enumerate(self.classes):  # for each class
            for maxDet_idx, maxDet in enumerate(self.max_detections):  # for each maximum number of detections
                results = [r[cls_idx] for r in results_list if cls_idx in r]

                if len(results) == 0:
                    logger.warning(f"WARNING, no results found for coco metric for class {cls_i}")
                    continue

                for no, r in enumerate(results):
                    dt_scores = np.concatenate([r['dtScores'][0:maxDet]])
                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dt_scores, kind='mergesort')
                    dt_scores_sorted = dt_scores[inds]

                    # r['dtMatches'] [T, R], where R = sum(all detections)
                    dt_matches = np.concatenate([r['dtMatches'][:, 0:maxDet]], axis=1)[:, inds]
                    dt_ignores = np.concatenate([r['dtIgnore'][:, 0:maxDet]], axis=1)[:, inds]
                    self.check_number_of_iou(dt_matches, dt_ignores)
                    gt_ignore = np.concatenate([r['gtIgnore']])
                    num_gt = np.count_nonzero(gt_ignore == 0)  # number of ground truth boxes (non ignored)
                    if num_gt == 0:
                        logger.warning(f"WARNING, no gt found for coco metric for class {cls_i}")
                        continue

                    # ignore cases need to be handled differently for tp and fp
                    tps = np.logical_and(dt_matches,  np.logical_not(dt_ignores))
                    fps = np.logical_and(np.logical_not(dt_matches), np.logical_not(dt_ignores))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float32)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float32)

                    for th_ind, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):  # for each threshold th_ind
                        tp, fp = np.array(tp), np.array(fp)
                        r, p, s = compute_stats_single_threshold(tp, fp, dt_scores_sorted, self.recall_thresholds, num_gt)
                        recall[no, th_ind, cls_idx, maxDet_idx] = r
                        precision[no, th_ind, :, cls_idx, maxDet_idx] = p
                        # corresponding score thresholds for recall steps
                        scores[no, th_ind, :, cls_idx, maxDet_idx] = s
        # pdb.set_trace()
        return {
            'counts': [num_iou_th, num_recall_th, num_classes, num_max_detections],  # [4]
            'recall':   recall,  # [num_iou_th, num_classes, num_max_detections]
            'precision': precision,  # [num_iou_th, num_recall_th, num_classes, num_max_detections]
            'scores': scores,  # [num_iou_th, num_recall_th, num_classes, num_max_detections]
        }


def compute_stats_single_threshold(tp: np.ndarray, fp: np.ndarray, dt_scores_sorted: np.ndarray, 
                                   recall_thresholds: Sequence[float], num_gt: int) -> Tuple[
                                       float, np.ndarray, np.ndarray]:
    """
    Compute recall value, precision curve and scores thresholds
    Adapted from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

    Args:
        tp (np.ndarray): cumsum over true positives [R], R is the number of detections
        fp (np.ndarray): cumsum over false positives [R], R is the number of detections
        dt_scores_sorted (np.ndarray): sorted (descending) scores [R], R is the number of detections
        recall_thresholds (Sequence[float]): recall thresholds which should be evaluated
        num_gt (int): number of ground truth bounding boxes (excluding boxes which are ignored)

    Returns:
        float: overall recall for given IoU value
        np.ndarray: precision values at defined recall values
            [RTH], where RTH is the number of recall thresholds
        np.ndarray: prediction scores corresponding to recall values
            [RTH], where RTH is the number of recall thresholds
    """
    num_recall_th = len(recall_thresholds)

    rc = tp / num_gt
    # np.spacing(1) is the smallest representable epsilon with float
    pr = tp / (fp + tp + np.spacing(1))

    if len(tp):
        recall = rc[-1]
    else:
        # no prediction
        recall = 0

    # array where precision values nearest to given recall th are saved
    precision = np.zeros((num_recall_th,))
    # save scores for corresponding recall value in here
    th_scores = np.zeros((num_recall_th,))
    # numpy is slow without cython optimization for accessing elements
    # use python array gets significant speed improvement
    pr = pr.tolist(); precision = precision.tolist()

    # smooth precision curve (create box shape)
    for i in range(len(tp) - 1, 0, -1):
        if pr[i] > pr[i-1]:
            pr[i-1] = pr[i]

    # get indices to nearest given recall threshold (nn interpolation!)
    inds = np.searchsorted(rc, recall_thresholds, side='left')
    try:
        for save_idx, array_index in enumerate(inds):
            precision[save_idx] = pr[array_index]
            th_scores[save_idx] = dt_scores_sorted[array_index]
    except:
        pass

    return recall, np.array(precision), np.array(th_scores)
