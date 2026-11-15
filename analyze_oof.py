"""Where does PQ go? Break out-of-fold errors down by cause.

  python analyze_oof.py --run runs/baseline [--eval-size 1024]

For every GT filament (per annotation set) the best-overlapping prediction
decides its fate:
  tp            IoU > 0.5
  near_miss     best IoU in (0.3, 0.5]: found, but the extent is wrong
  split         touched by >= 2 predictions, none > 0.5
  merged        its best prediction also touches another GT of the same set
  weak          detected poorly: best IoU in (0, 0.3]
  unseen        no prediction overlaps it; `max_prob` says whether the model
                saw it at all (below threshold) or post-processing dropped it
Each row also records size and *agreement*: the share of the frame's
annotators who also drew this filament (1 annotator of 3 -> 0.33). FNs on
low-agreement filaments are mostly unwinnable.

For predictions: is an FP real-but-unlabelled by this annotator (it overlaps
another annotator's GT), or pure noise?
Writes <run>/analysis/{gt_errors.csv,pred_errors.csv} and prints summaries.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.coco_io import gt_rles, load_observations, mask_to_rle, rle_to_mask
from src.evaluation import disk_mask_at, load_disks, load_probs, resize_probs
from src.metrics import iou_matrix
from src.postprocess import instances_from_probs
from src.utils import load_config
from src.resources import limit_resources

SIZE_BINS = [0, 300, 700, 1500, 3000, 1e9]  # native-resolution px


def main():
    limit_resources(gpu=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/baseline")
    ap.add_argument("--eval-size", type=int, default=1024)
    args = ap.parse_args()
    cfg = load_config(sorted(glob.glob(os.path.join(args.run, "fold*", "config_used.yaml")))[0])
    pp = cfg["postprocess"]
    tuned = os.path.join(args.run, "postprocess_tuned.yaml")
    if os.path.exists(tuned):
        pp = yaml.safe_load(open(tuned))["postprocess"]
    heads = cfg["model"]["heads"]
    obs = load_observations(cfg["data"]["train_json"])
    disks = load_disks(cfg["data"]["cache_dir"])
    S = args.eval_size
    scale, area_scale = S / 2048, (2048 / S) ** 2

    g_rows, p_rows = [], []
    for oof_dir in sorted(glob.glob(os.path.join(args.run, "fold*", "oof"))):
        ids = sorted({os.path.basename(p).rsplit("_", 1)[0]
                      for p in glob.glob(os.path.join(oof_dir, "*_mask.png"))})
        for oid in tqdm(ids, desc=oof_dir):
            probs = resize_probs(load_probs(oof_dir, oid, heads, as_uint8=True), S)
            insts = instances_from_probs(probs, disk_mask_at(disks.get(oid), S, pp["disk_margin"]), pp)
            pred = [mask_to_rle(t.mask) for t in insts]
            sets = gt_rles(obs[oid], scale)
            set_ids = list(sets)
            n_ann = len(set_ids)
            for sid in set_ids:
                g = sets[sid]
                m = iou_matrix(g, pred)                       # (n_gt, n_pred)
                others = [r for s2 in set_ids if s2 != sid for r in sets[s2]]
                agree = iou_matrix(g, others) if others else np.zeros((len(g), 0))
                gt_gt = iou_matrix(g, g)
                for i, r in enumerate(g):
                    mask = rle_to_mask(r) > 0
                    best = m[i].max() if m.shape[1] else 0.0
                    j = int(m[i].argmax()) if m.shape[1] else -1
                    touching = int((m[i] > 0).sum()) if m.shape[1] else 0
                    if best > 0.5:
                        fate = "tp"
                    elif touching >= 2:
                        fate = "split"
                    elif j >= 0 and best > 0 and (m[:, j] > 0).sum() >= 2:
                        fate = "merged"
                    elif best > 0.3:
                        fate = "near_miss"
                    elif best > 0:
                        fate = "weak"
                    else:
                        fate = "unseen"
                    # agreement: fraction of annotators (incl. this one) with an IoU>0.3 match
                    n_agree = 1 + sum(agree[i, k] > 0.3 for k in range(agree.shape[1]))
                    g_rows.append({
                        "obs_id": oid, "set_id": sid, "fate": fate, "best_iou": best,
                        "area": int(mask.sum() * area_scale),
                        "agreement": min(1.0, n_agree / n_ann) if n_ann > 1 else np.nan,
                        "max_prob": float(probs["mask"][mask].max()) if mask.any() else 0.0,
                        "mean_prob": float(probs["mask"][mask].mean()) if mask.any() else 0.0,
                        "n_ann": n_ann, "near_gt": int((gt_gt[i] > 0).sum() > 1),
                    })
                for j, t in enumerate(insts):
                    hit = m[:, j].max() if m.shape[0] else 0.0
                    other_hit = iou_matrix(others, [pred[j]]).max() if others else 0.0
                    p_rows.append({"obs_id": oid, "set_id": sid, "score": t.score,
                                   "area": int(t.area * area_scale), "best_iou": hit,
                                   "tp": hit > 0.5, "other_annotator_iou": other_hit})

    out = os.path.join(args.run, "analysis")
    os.makedirs(out, exist_ok=True)
    g, p = pd.DataFrame(g_rows), pd.DataFrame(p_rows)
    g.to_csv(os.path.join(out, "gt_errors.csv"), index=False)
    p.to_csv(os.path.join(out, "pred_errors.csv"), index=False)

    pd.set_option("display.width", 160)
    print(f"\nGT filaments: {len(g)}  (TP {int((g.fate == 'tp').sum())})")
    print(g.fate.value_counts(normalize=True).round(3).to_string())
    g["size"] = pd.cut(g.area, SIZE_BINS, labels=["<300", "300-700", "700-1.5k", "1.5-3k", ">3k"])
    print("\nfate by size (row %):")
    print(pd.crosstab(g["size"], g.fate, normalize="index").round(2).assign(n=g["size"].value_counts()))
    multi = g[g.n_ann > 1].copy()
    multi["agree"] = pd.cut(multi.agreement, [0, 0.4, 0.7, 1.01], labels=["1 of n", "some", "all"])
    print("\nfate by annotator agreement (multi-annotator frames):")
    print(pd.crosstab(multi.agree, multi.fate, normalize="index").round(2).assign(n=multi.agree.value_counts()))
    un = g[g.fate == "unseen"]
    print(f"\nunseen GT: max mask prob  <0.3: {(un.max_prob < 0.3).mean():.2f}  "
          f"0.3-thr: {((un.max_prob >= 0.3) & (un.max_prob < pp['threshold'])).mean():.2f}  "
          f">=thr (dropped by filters): {(un.max_prob >= pp['threshold']).mean():.2f}")
    fp = p[~p.tp]
    print(f"\npredictions: {len(p)} (per annotation set), FP {len(fp)}; of FPs, "
          f"matched by another annotator (IoU>0.5): {(fp.other_annotator_iou > 0.5).mean():.2f}, "
          f"partial overlap with GT: {(fp.best_iou > 0).mean():.2f}")
    p["score_bin"] = pd.cut(p.score, [0, 0.6, 0.7, 0.8, 0.9, 1.0])
    print("\nprecision by instance score:")
    print(p.groupby("score_bin", observed=True).agg(n=("tp", "size"), precision=("tp", "mean"),
                                                    mean_iou=("best_iou", "mean")).round(3))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
