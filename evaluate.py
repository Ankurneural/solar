"""Score a submission-format CSV against a GT CSV, like the organizers' notebook.

  python evaluate.py --pred runs/baseline/oof_submission.csv --gt data/cache/gt_train.csv
  python evaluate.py --pred preds.csv --gt data/cache/gt_train.csv --only-pred-images

GT rows are "<annotator>-<obs>_<k>", predictions "<obs>_<k>". Every annotation
set in the GT is scored (pass --only-pred-images to restrict to observations
that appear in the prediction file, e.g. when scoring one fold).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from pycocotools import mask as mask_util

from src.evaluation import read_rle_csv
from src.metrics import PQAccumulator
from src.resources import limit_resources


def main():
    limit_resources(gpu=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--only-pred-images", action="store_true")
    ap.add_argument("--images", default=None, help="text file of obs ids to restrict to")
    args = ap.parse_args()

    gt = read_rle_csv(args.gt)       # keyed by "<annotator>-<obs>"
    pred = read_rle_csv(args.pred)   # keyed by "<obs>"
    keep = None
    if args.images:
        keep = set(open(args.images).read().split())
    elif args.only_pred_images:
        keep = set(pred)

    acc = PQAccumulator()
    dice, best_iou = [], []
    for set_id, g in gt.items():
        oid = set_id.split("-", 1)[1]
        if keep is not None and oid not in keep:
            continue
        p = pred.get(oid, [])
        m = acc.update(g, p)
        if g:
            best_iou.extend(m.max(1).tolist() if p else [0.0] * len(g))
        dice.append(foreground_dice(g, p))
    m = acc.compute()
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}, indent=1))
    print("1:n (GT touched by n preds):", dict(sorted(acc.gt_degree.items())))
    print("n:1 (pred touching n GTs): ", dict(sorted(acc.pred_degree.items())))
    print(f"foreground Dice per annotation set (all filaments as one mask): mean {np.mean(dice):.4f}, "
          f"median {np.median(dice):.4f}")
    tp = np.asarray(acc.tp_ious)
    if tp.size:
        print("matched IoU (TP) quartiles:", np.percentile(tp, [25, 50, 75]).round(3).tolist())
    b = np.asarray(best_iou)
    edges = [0, 1e-9, 0.3, 0.5, 0.7, 0.9, 1.0001]
    hist = np.histogram(b, edges)[0] / max(1, b.size)
    print("GT best-IoU distribution:", {f"{lo:.1f}-{hi:.1f}" if lo else "0 (missed)": round(float(h), 4)
                                          for lo, hi, h in zip(edges[:-1], edges[1:], hist)})


def foreground_dice(gt: list[dict], pred: list[dict]) -> float:
    """Binary Dice of the union of all GT filaments vs the union of all predictions (1.0 if both empty)."""
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    g, p = mask_util.merge(gt), mask_util.merge(pred)
    inter = mask_util.area(mask_util.merge([g, p], intersect=True))
    return float(2 * inter / (mask_util.area(g) + mask_util.area(p)))


if __name__ == "__main__":
    main()
