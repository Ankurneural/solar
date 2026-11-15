"""Level-2 selection: score candidates by the quality of their *refined* mask.

  python tune_stack_l2.py --run runs/w11_all_768b

tune_stack.py selects with a blend of the GBM score (trained on the *candidate* mask's target) and the refiner score,
but the submission paints the *refined* mask, whose IoU with the annotators differs. Here, per fold (out-of-fold):
  1. GBM (folds != k) and refiner (fold k's own) scores, refined masks           [GPU, cached in <run>/l2_cache.pkl]
  2. target of the refined mask: mean over annotators of IoU*[IoU > 0.5]          [CPU]
  3. level-2 GBM on [candidate features, GBM score, refiner score, refined-mask area, area ratio], trained on the other folds
  4. tree selection + painting exactly as tune_stack.py, sweeping tau (and a mix with the old blend score)
Writes <run>/stack_l2_sweep.csv; compare with <run>/stack_sweep.csv (same candidates, same refiners).
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util
from sklearn.ensemble import HistGradientBoostingRegressor
from tqdm import tqdm

from src.candidates import select
from src.coco_io import gt_rles, load_observations, mask_to_rle
from src.metrics import IOU_THR, PQAccumulator
from src.refiner import load_refiners
from src.resources import limit_resources
from src.utils import load_config, setup_device
from train_refiner import CropDataset, _Node
from train_scorer import GBM
from tune_stack import TAUS, refined_crop, refiner_pass

L2_TAUS = [0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38]
MIX = [1.0, 0.7, 0.5]  # weight of the level-2 score vs the tune_stack blend score


def stage1(run, table, X, cfg, obs, device):
    """GBM OOF scores, refiner scores, and refined crops (as RLE of the crop) for candidates any setting could keep."""
    ds = CropDataset(table, cfg, obs, cfg["model"]["heads"], train=False)
    gbm, ref = np.zeros(len(table)), np.zeros(len(table))
    crops = {}
    for k in sorted(table.fold.unique()):
        va = (table.fold == k).to_numpy()
        gbm[va] = HistGradientBoostingRegressor(**GBM).fit(X[~va], table.y[~va]).predict(X[va])
        model = load_refiners([os.path.join(run, "refiner", f"fold{k}.pt")], device)[0]
        ds.geom = geom = model.geom
        idx = np.flatnonzero(va)
        s, m = refiner_pass(model, ds, idx, device)
        ref[idx] = s
        for n, j in enumerate(idx):
            if max(gbm[j], s[n]) > min(TAUS) * 0.5:
                y0, x0, c = refined_crop(table.loc[j], m[n], geom)
                if c.sum() >= 50:
                    crops[j] = (y0, x0, c.shape, mask_util.encode(np.asfortranarray(c.astype(np.uint8))))
        print(f"[fold {k}] {sum(1 for j in idx if j in crops)} refined crops", flush=True)
        del model
        torch.cuda.empty_cache()
    return gbm, ref, crops


def crop_full(c):
    y0, x0, shape, rle = c
    full = np.zeros((2048, 2048), bool)
    full[y0:y0 + shape[0], x0:x0 + shape[1]] = mask_util.decode(rle).astype(bool)
    return full


def refined_targets(table, crops, obs):
    """Mean over annotators of IoU*[IoU>0.5] for each refined mask."""
    y = np.full(len(table), np.nan)
    area = np.zeros(len(table))
    for oid, grp in tqdm(table.groupby("obs_id", sort=False), desc="refined targets"):
        js = [j for j in grp.index if j in crops]
        if not js:
            continue
        rles = [mask_to_rle(crop_full(crops[j])) for j in js]
        sets = gt_rles(obs[oid])
        vals = np.zeros((len(js), max(1, len(sets))))
        for si, g in enumerate(sets.values()):
            if g:
                iou = mask_util.iou(rles, g, [0] * len(g))
                best = np.asarray(iou).reshape(len(js), len(g)).max(1)
                vals[:, si] = np.where(best > IOU_THR, best, 0.0)
        y[js] = vals.mean(1)
        area[js] = mask_util.area(rles)
    return y, area


def sweep(table, crops, score_sets, obs):
    frames = list(table.groupby("obs_id", sort=False))
    gts = {oid: gt_rles(obs[oid]) for oid, _ in frames}
    rows = []
    for name, sc_all, tau in tqdm(score_sets, desc="sweep"):
        acc = PQAccumulator()
        for oid, grp in frames:
            pos = grp.index.to_numpy()
            nodes = [_Node(r.level, r.parent, r.children) for r in grp.itertuples()]
            sc = sc_all[pos]
            keep = sorted(select(nodes, sc, tau), key=lambda j: -sc[j])
            taken = np.zeros((2048, 2048), bool)
            pred = []
            for j in keep:
                g = pos[j]
                if g not in crops:
                    continue
                m = crop_full(crops[g]) & ~taken
                if m.sum() >= 50:
                    taken |= m
                    pred.append(mask_to_rle(m))
            for gset in gts[oid].values():
                acc.update(gset, pred)
        r = acc.compute()
        rows.append({"score": name, "tau": tau, **{k: r[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn")}})
        print(f"  {name:12s} tau {tau:.2f}: PQ {r['pq']:.4f}", flush=True)
    return pd.DataFrame(rows)


def main():
    limit_resources()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--blend", type=float, default=None, help="tune_stack blend for the reference score (default: stack.yaml)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N frames per fold")
    args = ap.parse_args()
    cfg = load_config(sorted(glob.glob(os.path.join(args.run, "fold*", "config_used.yaml")))[0])
    obs = load_observations(cfg["data"]["train_json"])
    rdir = os.path.join(args.run, "refiner")
    table = pd.read_pickle(os.path.join(rdir, "candidates.pkl")).reset_index(drop=True)
    X = np.load(os.path.join(rdir, "candidates.pkl.X.npy"))
    if args.limit:
        keep_ids = [o for _, a in table.groupby("fold").obs_id for o in a.unique()[:args.limit]]
        sub = table.obs_id.isin(keep_ids).to_numpy()
        X = X[sub]
        table = table[sub].reset_index(drop=True)  # tree links are positional within a frame, so they survive
    cache = os.path.join(args.run, "l2_cache_lim.pkl" if args.limit else "l2_cache.pkl")
    if os.path.exists(cache):
        gbm, ref, crops = pickle.load(open(cache, "rb"))
    else:
        gbm, ref, crops = stage1(args.run, table, X, cfg, obs, setup_device())
        pickle.dump((gbm, ref, crops), open(cache, "wb"))
    y_ref, area = refined_targets(table, crops, obs)
    has = ~np.isnan(y_ref)
    cand_area = table.rle.map(mask_util.area).to_numpy() * (2048 / table.rle.iloc[0]["size"][0]) ** 2
    F = np.column_stack([X, gbm, ref, np.log1p(area), area / np.maximum(cand_area, 1)])
    print(f"refined targets for {has.sum()} of {len(table)} candidates; mean y_ref {np.nanmean(y_ref):.3f} "
          f"vs candidate y {table.y[has].mean():.3f}")
    l2 = np.zeros(len(table))
    folds = table.fold.to_numpy()
    for k in np.unique(folds):
        tr, va = has & (folds != k), has & (folds == k)
        l2[va] = HistGradientBoostingRegressor(**GBM).fit(F[tr], y_ref[tr]).predict(F[va])
    print(f"corr with refined target: L2 {np.corrcoef(l2[has], y_ref[has])[0, 1]:.3f}  "
          f"GBM {np.corrcoef(gbm[has], y_ref[has])[0, 1]:.3f}  refiner {np.corrcoef(ref[has], y_ref[has])[0, 1]:.3f}")
    import yaml
    st = yaml.safe_load(open(os.path.join(args.run, "stack.yaml")))
    b = st["blend"] if args.blend is None else args.blend
    base = b * gbm + (1 - b) * ref
    sets = [(f"base_b{b}", base, st["tau"])]
    for w in MIX:
        sc = w * l2 + (1 - w) * base
        sets += [(f"l2_mix{w}", sc, t) for t in L2_TAUS]
    res = sweep(table, crops, sets, obs)
    res.to_csv(os.path.join(args.run, "stack_l2_sweep.csv"), index=False)
    print(res.pivot(index="tau", columns="score", values="pq").round(4).to_string())
    best = res.loc[res.pq.idxmax()]
    print(f"\nbest: {best.score} tau {best.tau} -> PQ {best.pq:.4f} (SQ {best.sq:.3f} RQ {best.rq:.3f}); "
          f"tune_stack: {st['oof_pq']:.4f}")


if __name__ == "__main__":
    main()
