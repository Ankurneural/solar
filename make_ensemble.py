"""Combine several runs into one ensemble "run" by averaging their OOF probabilities.

  python make_ensemble.py --runs runs/baseline runs/crop2048 --out runs/ens1 [--weights 0.4 0.6]

The output directory looks like a normal run (fold*/oof/*.png + config_used.yaml),
so tune_postprocess.py, analyze_oof.py and train_scorer.py work on it unchanged.
It also records the members in ensemble.yaml; infer.py then predicts with every
member's checkpoints and averages the same way. All members must share the
same folds and heads.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src.utils import load_config
from src.resources import limit_resources


def main():
    limit_resources(gpu=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=1024, help="resolution of the averaged maps")
    args = ap.parse_args()
    w = np.asarray(args.weights or [1.0] * len(args.runs), np.float32)
    w = w / w.sum()

    heads = None
    for r in args.runs:
        h = load_config(sorted(glob.glob(os.path.join(r, "fold*", "config_used.yaml")))[0])["model"]["heads"]
        if heads is not None and h != heads:
            raise SystemExit(f"{r} has heads {h}, expected {heads}")
        heads = h

    for fold_dir in sorted(glob.glob(os.path.join(args.runs[0], "fold*"))):
        fold = os.path.basename(fold_dir)
        out_oof = os.path.join(args.out, fold, "oof")
        os.makedirs(out_oof, exist_ok=True)
        shutil.copy(os.path.join(fold_dir, "config_used.yaml"), os.path.join(args.out, fold))
        names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(fold_dir, "oof", "*.png")))
        for name in tqdm(names, desc=fold):
            acc = np.zeros((args.size, args.size), np.float32)
            for r, wi in zip(args.runs, w):
                m = cv2.imread(os.path.join(r, fold, "oof", name), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    raise SystemExit(f"missing {r}/{fold}/oof/{name}")
                acc += wi * cv2.resize(m.astype(np.float32), (args.size, args.size),
                                       interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_oof, name), np.clip(acc + 0.5, 0, 255).astype(np.uint8))

    def checkpoint_run(r):  # OOF-only runs from predict_oof.py point at their source run
        src = os.path.join(r, "source.yaml")
        return yaml.safe_load(open(src))["run"] if os.path.exists(src) else r

    with open(os.path.join(args.out, "ensemble.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"members": [{"run": checkpoint_run(r), "weight": float(wi)}
                                    for r, wi in zip(args.runs, w)],
                        "oof_dirs": list(args.runs),  # per-member OOF maps (scorer features)
                        "size": args.size}, f, sort_keys=False)
    print(f"-> {args.out} ({len(args.runs)} members, weights {w.round(3).tolist()})")


if __name__ == "__main__":
    main()
