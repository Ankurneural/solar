"""Train the candidate scorer on out-of-fold predictions and measure its PQ.

  python train_scorer.py --run runs/baseline

1. Build candidate trees for every OOF frame (src/candidates.py) and their
   target: mean over annotators of IoU * [IoU > 0.5].
2. Cross-validate a gradient-boosted regressor with the SAME folds as the
   segmentation models (fold k's scorer never sees fold k's frames).
3. Tune tau (the keep threshold) on those honest predictions; compare PQ with
   the tuned single-threshold post-processing.
4. Fit on all candidates and save <run>/scorer.joblib for infer.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

from src.scorer_models import HitIoUModel, annotator_rows
from tqdm import tqdm

from src.candidates import feature_matrix, feature_names, generate, select
from src.coco_io import gt_rles, load_observations, mask_to_rle
from src.dataset import load_cached
from src.evaluation import add_member_maps, disk_mask_at, load_disks, load_probs, member_oof_dirs, resize_probs
from src.metrics import IOU_THR, iou_matrix
from src.utils import load_config
from src.resources import limit_resources

CAND = {"levels": [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "close_px": 0,
        "merge_px": 0, "min_area": 100}  # merge 0 > 10 > 20 px on ENS2 (docs/EXPERIMENTS.md)
GBM = dict(max_iter=500, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=100,
           l2_regularization=5.0, max_features=0.5, random_state=0)


def frame_inputs(cfg, oid, probs, disks, S):
    """Everything generate() needs at working size S."""
    import cv2

    img = cv2.resize(load_cached(cfg["data"]["cache_dir"], oid), (S, S), interpolation=cv2.INTER_AREA)
    dk = disks.get(oid)
    s = S / 2048
    disk_rc = (dk.cx * s, dk.cy * s, dk.r * s) if dk else None
    return img, disk_mask_at(dk, S, cfg["postprocess"].get("disk_margin", 0.01)), disk_rc


def pq_from_selection(frames, sel_per_frame) -> dict:
    tp_iou, fp, fn = [], 0, 0
    for fr, sel in zip(frames, sel_per_frame):
        for m in fr["iou"]:                     # one (n_gt, n_cand) matrix per annotator
            if m.shape[0] == 0:
                fp += len(sel)
                continue
            sub = m[:, sel] if len(sel) else np.zeros((m.shape[0], 0))
            hit = sub > IOU_THR
            tp_iou += sub[hit].tolist()
            fp += int((hit.sum(0) == 0).sum())
            fn += int((hit.sum(1) == 0).sum())
    tp = len(tp_iou)
    den = tp + 0.5 * fp + 0.5 * fn
    return {"pq": sum(tp_iou) / den if den else 0.0, "tp": tp, "fp": fp, "fn": fn,
            "sq": float(np.mean(tp_iou)) if tp else 0.0}


def main():
    limit_resources(gpu=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/baseline")
    ap.add_argument("--eval-size", type=int, default=1024)
    ap.add_argument("--reuse-cache", action="store_true",
                    help="reuse <run>/scorer_cache.pkl (candidates + targets) instead of rebuilding")
    ap.add_argument("--gbm", default="{}", help='JSON overrides, e.g. {"max_iter": 1000}')
    ap.add_argument("--cand", default="{}", help='candidate-tree overrides, e.g. {"merge_px": 0}')
    ap.add_argument("--tag", default="", help="suffix for output files, to compare variants side by side")
    ap.add_argument("--objective", default="reg", choices=["reg", "hitiou"],
                    help="reg: regress mean IoU*[IoU>0.5]; hitiou: P(match) x E[IoU|match] "
                         "(src/scorer_models.py)")
    args = ap.parse_args()
    gbm = {**GBM, **json.loads(args.gbm)}
    cand = {**CAND, **json.loads(args.cand)}
    tag = args.tag
    cfg = load_config(sorted(glob.glob(os.path.join(args.run, "fold*", "config_used.yaml")))[0])
    heads = cfg["model"]["heads"]
    obs = load_observations(cfg["data"]["train_json"])
    disks = load_disks(cfg["data"]["cache_dir"])
    folds = pd.read_csv(os.path.join(cfg["data"]["cache_dir"], "folds.csv")).set_index("obs_id").fold
    S, scale = args.eval_size, args.eval_size / 2048

    cache = os.path.join(args.run, f"scorer_cache{tag}.pkl")
    if args.reuse_cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            frames, X, y, g = pickle.load(f)
        print(f"loaded {cache}")
    else:
        frames, X, y, g = build_dataset(args.run, cfg, heads, obs, disks, folds, S, scale, cand)
        with open(cache, "wb") as f:
            pickle.dump((frames, X, y, g), f)
    print(f"{len(frames)} frames, {len(y)} candidates ({len(y) / len(frames):.0f}/frame), "
          f"positive {(y > 0).mean():.2f}")
    names = feature_names(len(member_oof_dirs(args.run)))
    if X.shape[1] != len(names):
        raise SystemExit("feature set changed since the cache was built; drop --reuse-cache")

    offs = np.cumsum([0] + [len(f["cands"]) for f in frames])
    if args.objective == "hitiou":
        hit, iou, row = annotator_rows(frames, offs)

    def fit(mask):
        """Scorer trained on the candidates where mask is True."""
        if args.objective == "reg":
            return HistGradientBoostingRegressor(**gbm).fit(X[mask], y[mask])
        keep = mask[row]
        sub = np.cumsum(mask) - 1  # global candidate index -> index within X[mask]
        return HitIoUModel(gbm).fit(X[mask], hit[keep], iou[keep], sub[row[keep]])

    # Honest OOF scores: one scorer per fold.
    pred = np.zeros_like(y)
    for k in np.unique(g):
        pred[g == k] = fit(g != k).predict(X[g == k])
    print(f"scorer OOF corr {np.corrcoef(pred, y)[0, 1]:.3f}")
    # Oracle: selection with the TRUE targets = ceiling of this candidate set.
    # A low ceiling means candidate generation (not scoring) is the bottleneck.
    for tau in (0.1, 0.2, 0.3):
        sels = [select(f["cands"], y[offs[i]:offs[i + 1]], tau) for i, f in enumerate(frames)]
        r = pq_from_selection(frames, sels)
        print(f"oracle tau={tau}: PQ {r['pq']:.4f} (TP {r['tp']} FP {r['fp']} FN {r['fn']})")
    results = []
    for tau in np.arange(0.06, 0.40, 0.02):
        sels = [select(f["cands"], pred[offs[i]:offs[i + 1]], tau) for i, f in enumerate(frames)]
        r = pq_from_selection(frames, sels)
        results.append({"tau": round(float(tau), 2), **r})
    res = pd.DataFrame(results)
    print(res.round(4).to_string(index=False))
    best = res.loc[res.pq.idxmax()]
    tuned = os.path.join(args.run, "postprocess_tuned.yaml")
    base = yaml.safe_load(open(tuned))["oof_metrics"]["pq"] if os.path.exists(tuned) else float("nan")
    print(f"\nlearned selection: PQ {best.pq:.4f} at tau={best.tau}  "
          f"(single-threshold tuned: {base:.4f}, delta {best.pq - base:+.4f})")

    final = fit(np.ones(len(y), bool))
    joblib.dump({"model": final, "tau": float(best.tau), "cand": cand, "features": names,
                 "oof_pq": float(best.pq), "eval_size": S, "gbm": gbm, "objective": args.objective},
                os.path.join(args.run, f"scorer{tag}.joblib"))
    res.to_csv(os.path.join(args.run, f"scorer_tau_sweep{tag}.csv"), index=False)
    print(f"-> {args.run}/scorer{tag}.joblib")


def build_dataset(run, cfg, heads, obs, disks, folds, S, scale, cand=CAND):
    frames, X, y, g = [], [], [], []
    t0 = time.time()
    mdirs = member_oof_dirs(run)
    names = feature_names(len(mdirs))
    for oof_dir in sorted(glob.glob(os.path.join(run, "fold*", "oof"))):
        fold = os.path.basename(os.path.dirname(oof_dir))
        ids = sorted({os.path.basename(p).rsplit("_", 1)[0]
                      for p in glob.glob(os.path.join(oof_dir, "*_mask.png"))})
        for oid in tqdm(ids, desc=f"candidates {oof_dir}"):
            probs = resize_probs(add_member_maps(load_probs(oof_dir, oid, heads, as_uint8=True),
                                                 mdirs, fold, oid, S), S)
            img, dmask, disk_rc = frame_inputs(cfg, oid, probs, disks, S)
            cands = generate(probs, dmask, img, disk_rc, cand, oid[-2:])
            rles = [mask_to_rle(c.full(S)) for c in cands]
            ious = [iou_matrix(gs, rles) for gs in gt_rles(obs[oid], scale).values()]
            # target: mean over annotators of (best IoU if > 0.5 else 0)
            tgt = np.mean([np.where(m.max(0) > IOU_THR, m.max(0), 0.0) if m.shape[0]
                           else np.zeros(len(cands)) for m in ious], axis=0) if cands else np.zeros(0)
            for c in cands:
                c.mask = None  # selection only needs the tree; keeps the cache small
            frames.append({"obs_id": oid, "cands": cands, "iou": ious})
            X.append(feature_matrix(cands, names))
            y.append(tgt)
            g.append(np.full(len(cands), folds[oid]))
    print(f"built candidates in {(time.time() - t0) / 60:.1f} min")
    return frames, np.concatenate(X), np.concatenate(y), np.concatenate(g)


if __name__ == "__main__":
    main()
