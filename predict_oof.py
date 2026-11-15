"""Regenerate out-of-fold probability maps for a trained run, e.g. with stronger TTA.

  python predict_oof.py --run runs/crop2048 --tta d4 --out runs/crop2048_d4

Why: the scorer and refiner learn from OOF maps, but test maps come from a
fold ensemble with D4 TTA, which is smoother. Matching the TTA at OOF time
removes part of that calibration gap (the fold-ensemble part can't be
reproduced honestly: only one model per fold never saw a given frame).
The output directory looks like a normal run (fold*/oof + config_used.yaml), so
make_ensemble.py, tune_postprocess.py, train_scorer.py etc. work on it unchanged.
It records the source run in source.yaml so an ensemble built from it still
points infer.py at the real checkpoints.
"""
from __future__ import annotations

import argparse
import os
import shutil

import cv2
import pandas as pd
import yaml
from tqdm import tqdm

from src.dataset import load_cached
from src.evaluation import save_probs
from src.inference import load_models, predict_probs
from src.utils import setup_device
from src.resources import limit_resources


def _done(oof_dir: str, oid: str, heads: list[str]) -> bool:
    """All head maps exist and decode (a crash can leave a truncated PNG)."""
    for h in heads:
        p = os.path.join(oof_dir, f"{oid}_{h}.png")
        if not os.path.exists(p) or cv2.imread(p, cv2.IMREAD_GRAYSCALE) is None:
            return False
    return True


def main():
    limit_resources(gpu=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tta", default="d4")
    ap.add_argument("--ckpt", default="best.pt")
    args = ap.parse_args()
    device = setup_device()
    for fold_dir in sorted(d for d in os.listdir(args.run) if d.startswith("fold")):
        src = os.path.join(args.run, fold_dir)
        if not os.path.exists(os.path.join(src, args.ckpt)):
            continue
        k = int(fold_dir[4:])
        models, cfg = load_models([os.path.join(src, args.ckpt)], device)
        cfg.setdefault("infer", {})["tta"] = args.tta
        folds = pd.read_csv(os.path.join(cfg["data"]["cache_dir"], "folds.csv"))
        out = os.path.join(args.out, fold_dir)
        os.makedirs(out, exist_ok=True)
        shutil.copy(os.path.join(src, "config_used.yaml"), out)
        heads = cfg["model"]["heads"]
        for oid in tqdm(folds.obs_id[folds.fold == k], desc=f"{fold_dir} ({args.tta})"):
            if _done(os.path.join(out, "oof"), oid, heads):  # resume after an interruption
                continue
            p = predict_probs(models, load_cached(cfg["data"]["cache_dir"], oid), cfg, device)
            save_probs(os.path.join(out, "oof"), oid, p)
    with open(os.path.join(args.out, "source.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"run": args.run, "tta": args.tta, "ckpt": args.ckpt}, f)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
