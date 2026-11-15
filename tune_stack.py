"""Tune the final selection stage on OOF: GBM/refiner score blend and keep threshold.

  python tune_stack.py --run runs/ens1

Uses the candidate table and per-fold refiners written by train_refiner.py
(each fold's candidates are scored by the GBM and the refiner that never saw
that fold). Refined masks are computed only for candidates some setting
selects, then cached. Scoring is exact (2048). Writes <run>/stack.yaml, which
infer.py reads for the blend weight and tau.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.candidates import select
from src.coco_io import gt_rles, load_observations, mask_to_rle
from src.evaluation import submission_rows, write_submission
from src.metrics import PQAccumulator
from pycocotools import mask as mask_util

from src.refiner import box_paste, largest_component, load_refiners, make_box, refine_batch
from src.resources import limit_resources, worker_budget, worker_init
from src.utils import load_config, setup_device
from train_refiner import CropDataset, _Node
from train_scorer import GBM

BLENDS = [1.0, 0.7, 0.5, 0.3, 0.15]
TAUS = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36]


def refiner_pass(models, ds, idx, device):
    """-> (scores for idx, {row: uint8 mask crop}) for the given table rows; several refiners are averaged."""
    models = models if isinstance(models, (list, tuple)) else [models]
    loader = DataLoader(Subset(ds, idx), batch_size=64, shuffle=False, num_workers=worker_budget(4),
                        worker_init_fn=worker_init)
    scores, masks = [], []
    for x, _, _ in tqdm(loader, desc="refiner", leave=False):
        outs = [refine_batch(mo, x.numpy(), device) for mo in models]
        m = np.mean([o[0] for o in outs], 0)
        s = np.mean([o[1] for o in outs], 0)
        scores.append(s)
        masks.append((m * 255).astype(np.uint8))
    return np.concatenate(scores), np.concatenate(masks)


def fold_refiner_paths(rdir: str, k: int) -> list[str]:
    """fold{k}.pt plus extra refiners of the same fold (fold{k}_<tag>.pt), e.g. other training seeds."""
    return [os.path.join(rdir, f"fold{k}.pt")] + sorted(glob.glob(os.path.join(rdir, f"fold{k}_*.pt")))


def refined_crop(row, crop_u8, geom):
    """Like src.refiner.refined_native but returns (y0, x0, bool crop): small enough to cache."""
    box = make_box(geom, row.bbox, mask_util.decode(row.rle) if geom != "square" else None)
    y0, x0, p = box_paste(crop_u8.astype(np.float32) / 255, box)
    return y0, x0, largest_component(p > 0.5)


def main():
    limit_resources()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/ens1")
    ap.add_argument("--write-oof", default=None,
                    help="read-only mode: select at <run>/stack.yaml's blend/tau, write the OOF predictions as a "
                         "submission-format CSV (for evaluate.py / the report); stack.yaml is not touched")
    args = ap.parse_args()
    device = setup_device()
    cfg = load_config(sorted(glob.glob(os.path.join(args.run, "fold*", "config_used.yaml")))[0])
    obs = load_observations(cfg["data"]["train_json"])
    rdir = os.path.join(args.run, "refiner")
    table = pd.read_pickle(os.path.join(rdir, "candidates.pkl")).reset_index(drop=True)
    X = np.load(os.path.join(rdir, "candidates.pkl.X.npy"))
    ds = CropDataset(table, cfg, obs, cfg["model"]["heads"], train=False)

    gbm, ref = np.zeros(len(table)), np.zeros(len(table))
    masks = {}
    for k in sorted(table.fold.unique()):
        va = (table.fold == k).to_numpy()
        gbm[va] = HistGradientBoostingRegressor(**GBM).fit(X[~va], table.y[~va]).predict(X[va])
        models = load_refiners(fold_refiner_paths(rdir, k), device)
        ds.geom = geom = models[0].geom
        # Refine every candidate that any (blend, tau) could keep: score above the lowest tau
        # under GBM alone, or any candidate (refiner score is needed for all of them).
        idx = np.flatnonzero(va)
        s, m = refiner_pass(models, ds, idx, device)
        ref[idx] = s
        keep_masks = idx[np.maximum(gbm[idx], s) > min(TAUS) * 0.8]
        pos = {j: n for n, j in enumerate(idx)}
        masks.update({j: m[pos[j]] for j in keep_masks})
        print(f"[fold {k}] {len(models)} refiner(s); corr GBM {np.corrcoef(gbm[va], table.y[va])[0, 1]:.3f} "
              f"refiner {np.corrcoef(ref[va], table.y[va])[0, 1]:.3f}")
        del models
        torch.cuda.empty_cache()

    frames = list(table.groupby("obs_id", sort=False))
    gts = {oid: gt_rles(obs[oid]) for oid, _ in frames}
    crops: dict[int, tuple] = {}
    rows = []
    grid = list(itertools.product(BLENDS, TAUS))
    if args.write_oof:
        st = yaml.safe_load(open(os.path.join(args.run, "stack.yaml")))
        grid = [(st["blend"], st["tau"])]
        sub_rows = []
    for b, tau in tqdm(grid, desc="blend x tau"):
        acc = PQAccumulator()
        for oid, grp in frames:
            pos = grp.index.to_numpy()
            nodes = [_Node(r.level, r.parent, r.children) for r in grp.itertuples()]
            sc = b * gbm[pos] + (1 - b) * ref[pos]
            keep = sorted(select(nodes, sc, tau), key=lambda j: -sc[j])
            taken = np.zeros((2048, 2048), bool)
            pred = []
            for j in keep:
                g = pos[j]
                if g not in masks:
                    continue
                if g not in crops:
                    crops[g] = refined_crop(grp.loc[g], masks[g], geom)
                y0, x0, m = crops[g]
                sub = taken[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
                mk = m & ~sub
                if mk.sum() >= 50:
                    sub |= mk
                    full = np.zeros((2048, 2048), bool)
                    full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = mk
                    pred.append(mask_to_rle(full))
            for gset in gts[oid].values():
                acc.update(gset, pred)
            if args.write_oof:
                sub_rows += submission_rows(oid, pred)
        r = acc.compute()
        rows.append({"blend": b, "tau": tau, **{k: r[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn")}})
    if args.write_oof:
        write_submission(sub_rows, args.write_oof)
        r = rows[0]
        print(f"OOF at blend {r['blend']} tau {r['tau']}: PQ {r['pq']:.4f} (SQ {r['sq']:.3f} RQ {r['rq']:.3f}) "
              f"-> {args.write_oof}")
        return
    res = pd.DataFrame(rows)
    print(res.pivot(index="tau", columns="blend", values="pq").round(4).to_string())
    best = res.loc[res.pq.idxmax()]
    print(f"\nbest: blend {best.blend} tau {best.tau} -> PQ {best.pq:.4f} "
          f"(SQ {best.sq:.3f} RQ {best.rq:.3f} TP {best.tp} FP {best.fp} FN {best.fn})")
    res.to_csv(os.path.join(args.run, "stack_sweep.csv"), index=False)
    with open(os.path.join(args.run, "stack.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"blend": float(best.blend), "tau": float(best.tau), "oof_pq": float(best.pq)},
                       f, sort_keys=False)
    print(f"-> {args.run}/stack.yaml")


if __name__ == "__main__":
    main()
