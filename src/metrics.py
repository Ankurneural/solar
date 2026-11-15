"""Panoptic Quality exactly as the organizers' Self-Evaluation notebook computes it.

Per annotation set (one annotator's labels of one observation):
  * IoU between every GT filament and every predicted filament of that observation
  * a pair is a hit if IoU > 0.5 (no Hungarian matching; with non-overlapping
    predictions at most one hit per GT/pred is possible, since IoU > 0.5 is unique)
  * TP = hits, FP = predictions hitting nothing, FN = GT hitting nothing
PQ = sum(IoU over TP) / (|TP| + 0.5|FP| + 0.5|FN|), pooled over ALL annotation sets.

Consequences worth remembering:
  * the same prediction is scored once per annotator -> optimise for consensus
  * every spurious blob costs 0.5 in the denominator regardless of its size
  * merging two GT filaments into one blob usually costs 2 FN + 1 FP
  * observations absent from GT (no filaments annotated) are never scored

We use pycocotools.mask.iou on RLEs, which is orders of magnitude faster than the
notebook's dense matmul and gives identical IoUs.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from pycocotools import mask as mask_util

IOU_THR = 0.5


def iou_matrix(gt: list[dict], pred: list[dict]) -> np.ndarray:
    """(n_gt, n_pred) IoU matrix between lists of COCO RLEs."""
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)), dtype=np.float64)
    return np.asarray(mask_util.iou(pred, gt, [0] * len(gt)), dtype=np.float64).T


@dataclass
class PQAccumulator:
    tp_ious: list = field(default_factory=list)
    fp: int = 0
    fn: int = 0
    n_sets: int = 0
    gt_degree: Counter = field(default_factory=Counter)    # 1:n relations (IoU > 0)
    pred_degree: Counter = field(default_factory=Counter)  # n:1 relations (IoU > 0)
    pair_ious: list = field(default_factory=list)          # all overlapping pairs (IoU > 0)

    def update(self, gt: list[dict], pred: list[dict]) -> np.ndarray:
        """Add one annotation set. Returns its IoU matrix."""
        self.n_sets += 1
        m = iou_matrix(gt, pred)
        if len(gt) == 0:
            self.fp += len(pred)
            self.pred_degree.update([0] * len(pred))
            return m
        if len(pred) == 0:
            self.fn += len(gt)
            self.gt_degree.update([0] * len(gt))
            return m
        hit = m > IOU_THR
        self.tp_ious.extend(m[hit].tolist())
        self.fp += int((hit.sum(0) == 0).sum())
        self.fn += int((hit.sum(1) == 0).sum())
        touch = m > 0
        self.gt_degree.update(touch.sum(1).tolist())
        self.pred_degree.update(touch.sum(0).tolist())
        self.pair_ious.extend(m[touch].tolist())
        return m

    def compute(self) -> dict:
        tp = len(self.tp_ious)
        denom = tp + 0.5 * self.fp + 0.5 * self.fn
        sq = float(np.mean(self.tp_ious)) if tp else 0.0
        rq = tp / denom if denom else 0.0
        pair = np.asarray(self.pair_ious)
        return {
            "pq": sum(self.tp_ious) / denom if denom else 0.0,
            "sq": sq,                      # mean IoU of matched pairs
            "rq": rq,                      # F1-like detection quality; pq = sq * rq
            "tp": tp, "fp": self.fp, "fn": self.fn, "n_sets": self.n_sets,
            "mean_pair_iou": float(pair.mean()) if pair.size else 0.0,
            "mean_pair_dice": float((2 * pair / (1 + pair)).mean()) if pair.size else 0.0,
            "gt_split_frac": _frac_over(self.gt_degree, 1),     # GT touched by >1 pred
            "pred_merge_frac": _frac_over(self.pred_degree, 1),  # pred touching >1 GT
        }


def _frac_over(c: Counter, k: int) -> float:
    n = sum(c.values())
    return sum(v for d, v in c.items() if d > k) / n if n else 0.0


def score_predictions(gt_sets: dict[str, dict[str, list[dict]]],
                      preds: dict[str, list[dict]]) -> dict:
    """gt_sets: {obs_id: {set_id: [RLE]}}, preds: {obs_id: [RLE]} -> PQ dict."""
    acc = PQAccumulator()
    for oid, sets in gt_sets.items():
        p = preds.get(oid, [])
        for g in sets.values():
            acc.update(g, p)
    return acc.compute()
