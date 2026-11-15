"""Train the stage-2 crop refiner per fold and measure what it adds to PQ.

  python train_refiner.py --run runs/baseline --folds 0          # quick check on one fold
  python train_refiner.py --run runs/baseline --folds all        # needed for honest tuning

Per fold k:
  1. candidates (src/candidates.py) of every OOF frame, with GBM scores from a
     scorer trained on the other folds (same as train_scorer.py)
  2. refiner trained on candidates of the other folds (src/refiner.py)
  3. on fold k, compare (exact PQ at 2048):
       gbm         GBM selection, candidate masks as they are
       gbm+mask    same selection, masks replaced by refined masks
       blend+mask  selection on a blend of GBM and refiner scores, refined masks
Outputs <run>/refiner/fold<k>.pt and <run>/refiner/report.csv.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.candidates import feature_matrix, feature_names, generate, select
from src.coco_io import gt_rles, load_observations, mask_to_rle
from src.dataset import load_cached
from src.evaluation import add_member_maps, load_disks, load_probs, member_oof_dirs, resize_probs
from src.metrics import IOU_THR, PQAccumulator, iou_matrix
from src.refiner import (ROT_H, ROT_W, box_inputs, build_refiner, crop_box, crop_inputs, jitter_affine,
                         make_box, refine_batch, refined_native, rot_box)
from src.resources import GpuThrottle, limit_resources, worker_budget, worker_init
from src.utils import load_config, seed_everything, setup_device
from train_scorer import CAND, GBM, frame_inputs

S = 1024            # candidate working size (as the scorer)
R = 256             # refiner crop size
LOADER_WORKERS = 8  # crop extraction is CPU-bound (--workers)
_SHIFT = 4


# ----------------------------------------------------------------- candidate table
def build_table(run, cfg, obs, disks, folds):
    heads = cfg["model"]["heads"]
    rows, X = [], []
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
            cands = generate(probs, dmask, img, disk_rc, CAND, oid[-2:])
            if not cands:
                continue
            rles = [mask_to_rle(c.full(S)) for c in cands]
            sets = gt_rles(obs[oid], S / 2048)
            per_set = {sid: iou_matrix(g, rles) for sid, g in sets.items()}
            X.append(feature_matrix(cands, names))
            for i, c in enumerate(cands):
                h, w = c.mask.shape
                best, match = [], {}
                for sid, m in per_set.items():
                    v = m[:, i].max() if m.shape[0] else 0.0
                    best.append(v if v > IOU_THR else 0.0)
                    if m.shape[0] and v > 0.1:
                        match[sid] = int(m[:, i].argmax())
                rows.append({"obs_id": oid, "fold": int(folds[oid]), "oof_dir": oof_dir, "idx": i,
                             "parent": c.parent, "level": c.level,
                             "children": list(c.children), "rle": rles[i],
                             "bbox": (c.x0 * 2048 / S, c.y0 * 2048 / S, (c.x0 + w) * 2048 / S,
                                      (c.y0 + h) * 2048 / S),
                             "y": float(np.mean(best)) if best else 0.0, "match": match})
    return pd.DataFrame(rows), np.concatenate(X)


# ----------------------------------------------------------------- dataset
class CropDataset(Dataset):
    def __init__(self, table: pd.DataFrame, cfg, obs, heads, train: bool, geom: str = "square"):
        self.t = table.reset_index(drop=True)
        self.cfg, self.obs, self.heads, self.train, self.geom = cfg, obs, heads, train, geom
        self._cache = {}

    def __len__(self):
        return len(self.t)

    def _frame(self, oid, oof_dir):
        if oid not in self._cache:
            if len(self._cache) > 8:
                self._cache.pop(next(iter(self._cache)))
            img = load_cached(self.cfg["data"]["cache_dir"], oid)
            prob = cv2.imread(os.path.join(oof_dir, f"{oid}_mask.png"), cv2.IMREAD_GRAYSCALE)
            self._cache[oid] = (img, prob)
        return self._cache[oid]

    def __getitem__(self, i):
        r = self.t.iloc[i]
        img, prob = self._frame(r.obs_id, r.oof_dir)
        seed = mask_util.decode(r.rle)
        o = self.obs[r.obs_id]
        n_sets = max(1, len(o.annotations))
        polys = [o.annotations[sid][gi].polygon for sid, gi in r.match.items()]
        if self.geom == "square":
            x, tgt = self._square(r, img, prob, seed, polys)
        else:
            A = rot_box(seed)
            if self.train:
                A = jitter_affine(A)
            x = box_inputs(img, prob, seed, A)
            tgt = np.zeros((ROT_H, ROT_W), np.float32)
            for poly in polys:  # COCO vertices are pixel edges; cv2 works in pixel centres
                pts = cv2.transform((poly - 0.5).reshape(-1, 1, 2).astype(np.float64), A)
                layer = np.zeros((ROT_H, ROT_W), np.uint8)
                cv2.fillPoly(layer, [(pts * (1 << _SHIFT)).round().astype(np.int32)], 1, shift=_SHIFT)
                tgt += layer
        tgt /= n_sets
        if self.train:
            if random.random() < 0.5:
                x, tgt = x[:, :, ::-1], tgt[:, ::-1]
            if random.random() < 0.5:
                x, tgt = x[:, ::-1], tgt[::-1]
            if self.geom == "square" and random.random() < 0.5:
                x, tgt = x.transpose(0, 2, 1), tgt.T
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(tgt)),
                torch.tensor(r.y, dtype=torch.float32))

    def _square(self, r, img, prob, seed, polys):
        cx, cy, side = crop_box(r.bbox)
        if self.train:  # jitter the window: the refiner must not rely on exact centring
            cx += int(np.random.uniform(-0.1, 0.1) * side)
            cy += int(np.random.uniform(-0.1, 0.1) * side)
            side = int(side * np.random.uniform(0.9, 1.15))
        x = crop_inputs(img, prob, seed, cx, cy, side, R)
        tgt = np.zeros((R, R), np.float32)
        k = R / side
        for poly in polys:  # COCO vertices are pixel edges; cv2 (and cv2.resize) work in pixel centres
            pts = (((poly - [cx - side // 2, cy - side // 2]) * k - 0.5) * (1 << _SHIFT)).round().astype(np.int32)
            layer = np.zeros((R, R), np.uint8)
            cv2.fillPoly(layer, [pts.reshape(-1, 1, 2)], 1, shift=_SHIFT)
            tgt += layer
        return x, tgt


# ----------------------------------------------------------------- train / predict
def train_one(table, cfg, obs, heads, device, epochs, samples_per_epoch, out_path,
              encoder="resnet18", lr=3e-4, geom="square"):
    model = build_refiner(encoder).to(device)
    model.geom = geom
    if os.path.exists(out_path):
        ck = torch.load(out_path, map_location=device)
        model.load_state_dict(ck["model"])
        model.geom = ck.get("geom", "square")
        print(f"  reusing finished {out_path}")
        return model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (samples_per_epoch // 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    ds = CropDataset(table, cfg, obs, heads, train=True, geom=geom)
    last_path, start = out_path + ".last", 0
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start = ck["epoch"]
        print(f"  resuming from epoch {start + 1}")
    for ep in range(start, epochs):
        idx = np.random.choice(len(ds), min(samples_per_epoch, len(ds)), replace=False)
        loader = DataLoader(torch.utils.data.Subset(ds, idx), batch_size=32, shuffle=True,
                            num_workers=worker_budget(LOADER_WORKERS), drop_last=True, persistent_workers=False,
                            worker_init_fn=worker_init)
        throttle = GpuThrottle()
        model.train()
        tot, n = 0.0, 0
        for x, t, y in tqdm(loader, desc=f"refiner epoch {ep + 1}/{epochs}", leave=False):
            throttle.begin()
            x, t, y = x.to(device), t.to(device), y.to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                m, s = model(x)
            m, s = m.float()[:, 0], s.float()[:, 0]
            p = torch.sigmoid(m)
            dice = 1 - (2 * (p * t).sum((1, 2)) + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)
            loss = (F.binary_cross_entropy_with_logits(m, t) + 0.5 * dice.mean()
                    + F.binary_cross_entropy_with_logits(s, y))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
            throttle.step(sync=False)
        print(f"  epoch {ep + 1}: loss {tot / max(1, n):.4f}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep + 1}, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)
    torch.save({"model": model.state_dict(), "encoder": encoder, "geom": geom}, out_path)
    os.remove(last_path)
    model.geom = geom
    return model


def predict_table(model, table, cfg, obs, heads, device):
    ds = CropDataset(table, cfg, obs, heads, train=False, geom=model.geom)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=worker_budget(LOADER_WORKERS),
                        worker_init_fn=worker_init)
    model.eval()
    masks, scores = [], []
    for x, _, _ in tqdm(loader, desc="refiner predict", leave=False):
        m, s = refine_batch(model, x.numpy(), device)
        masks.append((m * 255).astype(np.uint8))
        scores.append(s)
    return np.concatenate(masks), np.concatenate(scores)


# ----------------------------------------------------------------- evaluation
class _Node:  # minimal stand-in for Candidate in select()
    def __init__(self, level, parent, children):
        self.level, self.parent, self.children = level, parent, children


def refined_full(row, crop_prob_u8, geom) -> np.ndarray:
    box = make_box(geom, row.bbox, mask_util.decode(row.rle) if geom != "square" else None)
    return refined_native(crop_prob_u8.astype(np.float32) / 255, box)


def evaluate_fold(tab_k, gbm_pred, ref_masks, ref_scores, obs, taus, blends, geom="square"):
    """Exact PQ at 2048 for the three variants; returns list of result dicts.
    All arrays are aligned with the rows of tab_k."""
    out = []
    tab_k = tab_k.reset_index(drop=True)
    frames = list(tab_k.groupby("obs_id", sort=False))
    gt = {oid: gt_rles(obs[oid]) for oid, _ in frames}
    variants = [("gbm", 1.0, False), ("gbm+mask", 1.0, True)] + \
               [(f"blend{b}+mask", b, True) for b in blends]
    for name, b, use_ref_mask in variants:
        for tau in taus:
            acc = PQAccumulator()
            for oid, grp in frames:
                pos = grp.index.to_numpy()
                nodes = [_Node(r.level, r.parent, r.children) for r in grp.itertuples()]
                sc = b * gbm_pred[pos] + (1 - b) * ref_scores[pos]
                keep = sorted(select(nodes, sc, tau), key=lambda j: -sc[j])
                taken = np.zeros((2048, 2048), bool)
                pred = []
                for j in keep:
                    r = grp.iloc[j]
                    if use_ref_mask:
                        m = refined_full(r, ref_masks[pos[j]], geom)
                    else:
                        m = cv2.resize(mask_util.decode(r.rle).astype(np.float32), (2048, 2048),
                                       interpolation=cv2.INTER_LINEAR) > 0.5
                    m &= ~taken
                    if m.sum() >= 50:
                        taken |= m
                        pred.append(mask_to_rle(m))
                for g in gt[oid].values():
                    acc.update(g, pred)
            r = acc.compute()
            out.append({"variant": name, "tau": tau, **{k: r[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn")}})
            print(f"  {name:14s} tau={tau:.2f}: PQ {r['pq']:.4f} SQ {r['sq']:.3f} RQ {r['rq']:.3f}")
    return out


def main():
    global LOADER_WORKERS
    limit_resources()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/baseline")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--encoder", default="resnet18", help="smp encoder (non-HF weights: resnet*, efficientnet-b*)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=LOADER_WORKERS, help="DataLoader workers (capped by SF_CPU_FRAC)")
    ap.add_argument("--seed", type=int, default=0, help="training seed (refiner ensembles: train with another seed and --tag)")
    ap.add_argument("--geom", default="square", choices=["square", "rot"],
                    help="crop geometry: axis-aligned square, or rotated to the candidate's principal axis")
    ap.add_argument("--tag", default="", help="refiners go to <run>/refiner<tag>/ (the candidate table stays shared)")
    ap.add_argument("--samples-per-epoch", type=int, default=16000)
    ap.add_argument("--table-only", action="store_true",
                    help="only build <run>/refiner/candidates.pkl (e.g. to reuse refiners "
                         "trained on a closely related ensemble with tune_stack.py)")
    args = ap.parse_args()
    LOADER_WORKERS = args.workers
    seed_everything(args.seed)
    device = setup_device()
    cfg = load_config(sorted(glob.glob(os.path.join(args.run, "fold*", "config_used.yaml")))[0])
    heads = cfg["model"]["heads"]
    obs = load_observations(cfg["data"]["train_json"])
    disks = load_disks(cfg["data"]["cache_dir"])
    folds = pd.read_csv(os.path.join(cfg["data"]["cache_dir"], "folds.csv")).set_index("obs_id").fold
    table_dir = os.path.join(args.run, "refiner")
    out_dir = os.path.join(args.run, f"refiner{args.tag}")
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    tab_path = os.path.join(table_dir, "candidates.pkl")
    if os.path.exists(tab_path):
        table, X = pd.read_pickle(tab_path), np.load(tab_path + ".X.npy")
    else:
        t0 = time.time()
        table, X = build_table(args.run, cfg, obs, disks, folds)
        table.to_pickle(tab_path)
        np.save(tab_path + ".X.npy", X)
        print(f"{len(table)} candidates in {(time.time() - t0) / 60:.1f} min")

    if args.table_only:
        print(f"-> {tab_path}")
        return
    fold_list = sorted(table.fold.unique()) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    report = []
    for k in fold_list:
        tr, va = table.fold != k, table.fold == k
        gbm = HistGradientBoostingRegressor(**GBM).fit(X[tr.to_numpy()], table.y[tr])
        gbm_pred = np.zeros(len(table))
        gbm_pred[va.to_numpy()] = gbm.predict(X[va.to_numpy()])
        print(f"[fold {k}] refiner train on {tr.sum()} candidates")
        model = train_one(table[tr], cfg, obs, heads, device, args.epochs, args.samples_per_epoch,
                          os.path.join(out_dir, f"fold{k}.pt"), args.encoder, args.lr, args.geom)
        tab_k = table[va]
        masks, scores = predict_table(model, tab_k, cfg, obs, heads, device)
        gk, yk = gbm_pred[va.to_numpy()], tab_k.y.to_numpy()
        print(f"  score corr with target: GBM {np.corrcoef(gk, yk)[0, 1]:.3f}  "
              f"refiner {np.corrcoef(scores, yk)[0, 1]:.3f}  "
              f"blend {np.corrcoef(gk + scores, yk)[0, 1]:.3f}")
        res = evaluate_fold(tab_k, gk, masks, scores, obs,
                            taus=[0.14, 0.18, 0.22], blends=[0.5, 0.0], geom=model.geom)
        report += [{"fold": k, **r} for r in res]
        # keep rows of folds trained in earlier runs (resuming after an interruption)
        rep_path = os.path.join(out_dir, "report.csv")
        old_rows = pd.read_csv(rep_path) if os.path.exists(rep_path) else pd.DataFrame()
        if len(old_rows):
            old_rows = old_rows[~old_rows.fold.isin([r["fold"] for r in report])]
        pd.concat([old_rows, pd.DataFrame(report)]).sort_values(["fold", "variant", "tau"]).to_csv(
            rep_path, index=False)
    print(pd.DataFrame(report).groupby("variant").pq.max().round(4))


if __name__ == "__main__":
    main()
