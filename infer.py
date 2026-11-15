"""Predict filaments for a folder of frames and write the submission CSV.

  python infer.py --run runs/baseline                       # all folds, test images
  python infer.py --run runs/baseline --folds 0 --images <dir> --out preds.csv

Uses <run>/postprocess_tuned.yaml when present, else the config's defaults.
Images are preprocessed on the fly exactly as in prepare_data.py, so this works
on any folder of raw GONG JPEGs (only the supplied test image is used per
prediction, as the rules require).
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from prepare_data import list_images
from src.candidates import predict_with_scorer
from src.coco_io import mask_to_rle, obs_id_from_name
from src.evaluation import (predict_instances_rle, resize_probs, submission_rows,
                            write_submission)
from src.inference import load_models, predict_probs
from src.refiner import load_refiners
from src.solar_disk import preprocess
from src.utils import load_config, setup_device
from src.resources import limit_resources


def cache_key(run: str, paths: list[str], mcfg: dict, size) -> str:
    """Directory name for one member's cached test maps: run, TTA, map size and a hash of the
    checkpoint files (name/size/mtime), so retrained checkpoints never reuse stale maps."""
    import hashlib

    tta = (mcfg.get("infer") or {}).get("tta", "hflip")  # predict_probs default
    sig = "|".join(f"{p}:{os.path.getsize(p)}:{int(os.path.getmtime(p))}" for p in paths)
    h = hashlib.sha1(sig.encode()).hexdigest()[:10]
    return f"{os.path.basename(os.path.normpath(run))}__{tta}__{size or 'native'}__{h}"


def member_probs(models, mcfg, img, device, size, cache_dir, oid) -> dict:
    """Stage-1 maps of one member (resized to `size`), cached per frame as float16.
    Maps always round-trip through float16, so cached and fresh runs give identical submissions."""
    path = os.path.join(cache_dir, f"{oid}.npz") if cache_dir else None
    if path and os.path.exists(path):
        try:
            with np.load(path) as z:
                return {k: z[k].astype(np.float32) for k in z.files}
        except (OSError, ValueError):  # truncated file from a crash: recompute
            pass
    p = predict_probs(models, img, mcfg, device)
    if size:
        p = resize_probs(p, size)
    p = {k: np.asarray(v, np.float16) for k, v in p.items()}
    if path:
        tmp = path[:-4] + ".tmp.npz"
        np.savez(tmp, **p)
        os.replace(tmp, path)
    return {k: v.astype(np.float32) for k, v in p.items()}


def main():
    limit_resources(gpu=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir (default: paths.out_dir of --config)")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--folds", default="all", help="'all' or comma list, e.g. 0,2")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--images", default=None, help="raw image dir (default: config test_images)")
    ap.add_argument("--out", default=None, help="CSV path (default: <run>/submission.csv)")
    ap.add_argument("--tta", default=None, help="none | hflip | flip4 | d4")
    ap.add_argument("--no-scorer", action="store_true",
                    help="ignore <run>/scorer.joblib and use threshold post-processing")
    ap.add_argument("--no-refiner", action="store_true",
                    help="ignore <run>/refiner/fold*.pt (crop refiner of kept candidates)")
    ap.add_argument("--cache-dir", default="runs/test_cache",
                    help="per-member stage-1 test maps are cached here (float16), so variants of the "
                         "selection/refiner/ensemble weights skip stage 1; 'none' disables")
    ap.add_argument("--tau", type=float, default=None, help="override the keep threshold of stack.yaml / scorer")
    ap.add_argument("--blend", type=float, default=None, help="override the GBM/refiner score blend of stack.yaml")
    ap.add_argument("--refiner-dir", default=None, help="dir with the refiner fold*.pt (default: <run>/refiner)")
    args = ap.parse_args()
    args.run = args.run or load_config(args.config)["paths"]["out_dir"]
    cache_root = None if args.cache_dir.lower() == "none" else args.cache_dir

    device = setup_device()
    # A run made by make_ensemble.py lists member runs; each predicts with its own config.
    ens_path = os.path.join(args.run, "ensemble.yaml")
    ens = yaml.safe_load(open(ens_path)) if os.path.exists(ens_path) else None
    member_runs = [(m["run"], m["weight"]) for m in ens["members"]] if ens else [(args.run, 1.0)]
    members = []
    for run, weight in member_runs:
        paths = sorted(glob.glob(os.path.join(run, "fold*", args.ckpt)))
        if args.folds != "all":
            keep = {f"fold{int(f)}" for f in args.folds.split(",")}
            paths = [p for p in paths if os.path.basename(os.path.dirname(p)) in keep]
        if not paths:
            raise SystemExit(f"no checkpoints {args.ckpt} under {run}")
        models, mcfg = load_models(paths, device)
        if args.tta:
            mcfg.setdefault("infer", {})["tta"] = args.tta
        cdir = None
        if cache_root:
            cdir = os.path.join(cache_root, cache_key(run, paths, mcfg, ens["size"] if ens else None))
            os.makedirs(cdir, exist_ok=True)
        members.append((models, mcfg, weight, cdir))
    cfg = members[0][1]
    models = [m for mem in members for m in mem[0]]
    pp = cfg["postprocess"]
    tuned = os.path.join(args.run, "postprocess_tuned.yaml")
    if os.path.exists(tuned):
        with open(tuned, encoding="utf-8") as f:
            pp = yaml.safe_load(f)["postprocess"]
    scorer_path = os.path.join(args.run, "scorer.joblib")
    scorer = joblib.load(scorer_path) if os.path.exists(scorer_path) and not args.no_scorer else None
    ref_paths = sorted(glob.glob(os.path.join(args.refiner_dir or os.path.join(args.run, "refiner"), "fold*.pt")))
    refiners = load_refiners(ref_paths, device) if scorer and ref_paths and not args.no_refiner else None
    stack_path = os.path.join(args.run, "stack.yaml")
    stack = yaml.safe_load(open(stack_path)) if refiners and os.path.exists(stack_path) else {}
    blend, tau = stack.get("blend", 1.0), stack.get("tau")
    if args.blend is not None:
        blend = args.blend
    if args.tau is not None:
        tau = args.tau
    if scorer:
        print(f"{len(models)} model(s); learned selection (tau={tau or scorer['tau']}, "
              f"blend={blend}); {len(refiners or [])} refiner(s)"
              + (f"; stack OOF PQ {stack['oof_pq']:.4f}" if stack else ""))
    else:
        print(f"{len(models)} model(s); post-process: {pp}")

    img_dir = args.images or cfg["data"]["test_images"]
    out = args.out or os.path.join(args.run, "submission.csv")
    # Per-frame checkpoints: an interrupted run (restart, crash) resumes where it stopped.
    part_dir = out + ".parts"
    os.makedirs(part_dir, exist_ok=True)
    rows, t0, n_new = [], time.time(), 0
    for path in tqdm(list_images(img_dir), desc="infer"):
        oid = obs_id_from_name(path)
        part = os.path.join(part_dir, f"{oid}.csv")
        if os.path.exists(part):
            done = pd.read_csv(part, dtype=str, keep_default_na=False)
            rows += list(done.itertuples(index=False, name=None))
            continue
        raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img, disk = preprocess(raw, clahe_clip=cfg["data"].get("clahe_clip", 0.0),
                               use_prior=cfg["data"].get("disk_prior", True))
        if ens:
            probs = {}
            for mi, (mm, mcfg, wgt, cdir) in enumerate(members):
                mp = member_probs(mm, mcfg, img, device, ens["size"], cdir, oid)
                for h, v in mp.items():
                    probs[h] = probs.get(h, 0) + wgt * v
                if ens.get("member_features"):  # same uint8 quantisation as the OOF member maps
                    probs[f"member{mi}"] = np.round(mp["mask"] * 255) / 255
        else:
            probs = member_probs(members[0][0], cfg, img, device, None, members[0][3], oid)
        if scorer:
            masks = predict_with_scorer(probs, img, disk, scorer, oid[-2:], raw.shape[0],
                                        pp.get("disk_margin", 0.01), refiners, device, blend, tau)
            rles = [mask_to_rle(m) for m in masks]
        else:
            rles = predict_instances_rle(probs, disk, pp, raw.shape[0])
        new_rows = submission_rows(oid, rles)
        write_submission(new_rows, part + ".tmp")
        os.replace(part + ".tmp", part)  # atomic: a crash never leaves a half-written part
        rows += new_rows
        n_new += 1
    n_img = len(list_images(img_dir))
    write_submission(rows, out)
    dt = time.time() - t0
    print(f"{len(rows)} filaments over {n_img} images -> {out} "
          f"({n_new} predicted now, {n_img - n_new} resumed; "
          f"{dt / max(1, n_new):.2f}s per predicted image"
          f"{', GPU' if torch.cuda.is_available() else ''})")


if __name__ == "__main__":
    main()
