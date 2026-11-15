"""Time each inference stage on a few test frames.

  python tools/profile_infer.py --run runs/w71_b1 --n 3

Prints seconds per frame for: preprocessing, each stage-1 member (and each TTA
mode), candidate tree + features, GBM, refiners, and mask painting + RLE. It runs
under the usual resource caps (GPU duty throttle included), like real inference.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import joblib
import numpy as np
import torch
import yaml

from prepare_data import list_images
from src.candidates import feature_matrix, generate, select
from src.evaluation import disk_mask_at, resize_probs
from src.inference import load_models, predict_probs
from src.refiner import load_refiners, refined_native, run_refiners
from src.resources import limit_resources
from src.solar_disk import preprocess
from src.utils import setup_device


def tick(device):
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def main():
    limit_resources()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/w71_b1")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    device = setup_device()
    ens = yaml.safe_load(open(os.path.join(args.run, "ensemble.yaml")))
    members = []
    for m in ens["members"]:
        models, cfg = load_models(sorted(glob.glob(os.path.join(m["run"], "fold*", "best.pt"))), device)
        members.append((m["run"], models, cfg, m["weight"]))
    scorer = joblib.load(os.path.join(args.run, "scorer.joblib"))
    stack = yaml.safe_load(open(os.path.join(args.run, "stack.yaml")))
    refiners = load_refiners(sorted(glob.glob(os.path.join(args.run, "refiner", "fold*.pt"))), device)
    S = scorer["eval_size"]
    t = defaultdict(float)
    paths = list_images(members[0][2]["data"]["test_images"])[: args.n + 1]
    for fi, path in enumerate(paths):
        if fi == 1:  # frame 0 is a warm-up (cuDNN autotuning, lazy init): not counted
            t.clear()
        t0 = tick(device)
        raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img, disk = preprocess(raw)
        t["preprocess"] += tick(device) - t0
        probs = {}
        for run, models, cfg, w in members:
            for tta in ("hflip", "d4")[::-1]:
                cfg.setdefault("infer", {})["tta"] = tta
                t0 = tick(device)
                mp = predict_probs(models, img, cfg, device)
                t[f"{os.path.basename(run)} ({len(models)} folds, {tta})"] += tick(device) - t0
            for h, v in resize_probs(mp, S).items():
                probs[h] = probs.get(h, 0) + w * v
        t0 = tick(device)
        s = S / 2048
        cands = generate(probs, disk_mask_at(disk, S, 0.01), cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA),
                         (disk.cx * s, disk.cy * s, disk.r * s), scorer["cand"], "")
        t["candidates + features"] += tick(device) - t0
        t0 = tick(device)
        sc = scorer["model"].predict(feature_matrix(cands, scorer.get("features")))
        t["GBM"] += tick(device) - t0
        t0 = tick(device)
        k = 2048 / S
        bb = [(c.x0 * k, c.y0 * k, (c.x0 + c.mask.shape[1]) * k, (c.y0 + c.mask.shape[0]) * k) for c in cands]
        crops, rs, boxes = run_refiners(refiners, img, probs["mask"], [c.full(S) for c in cands], bb, device)
        t[f"refiners ({len(refiners)} models)"] += tick(device) - t0
        t["n_candidates"] += len(cands)
        t0 = tick(device)
        b = stack["blend"]
        keep = select(cands, b * sc + (1 - b) * rs, stack["tau"])
        _ = [refined_native(crops[i], boxes[i]) for i in keep]
        t["select + paint"] += tick(device) - t0
    n = len(paths) - 1
    total_d4 = sum(v for k, v in t.items() if "hflip" not in k and k != "n_candidates") / n
    print(f"\nper frame over {n} frames (GPU duty cap {os.environ.get('SF_GPU_DUTY', '0.65')}):")
    for k, v in t.items():
        print(f"  {k:55s} {v / n:7.2f} s")
    print(f"  {'TOTAL as deployed (D4 members)':55s} {total_d4:7.2f} s")


if __name__ == "__main__":
    main()
