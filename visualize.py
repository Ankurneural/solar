"""Overlay predictions on frames for qualitative review (and the report).

  python visualize.py --run runs/baseline --n 12                 # OOF frames vs GT
  python visualize.py --run runs/baseline --csv runs/baseline/submission.csv --images <test dir>

OOF mode draws, on the preprocessed frame:
  thin coloured outlines  ground truth, one colour per annotator
  filled green / red      predicted filaments matched (IoU > 0.5 with any annotator) / unmatched
Predictions come from the run's selection (scorer if present, else tuned
post-processing) applied to the OOF probabilities. The scorer was fitted on all
folds, so these OOF pictures are slightly optimistic; they are for looking, not scoring.
CSV mode draws a submission file's masks on raw test frames.
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import joblib
import numpy as np
import pandas as pd
import yaml

from src.candidates import predict_with_scorer
from src.coco_io import counts_to_rle, gt_rles, load_observations, mask_to_rle, rle_to_mask
from src.dataset import load_cached
from src.evaluation import load_disks, load_probs, predict_instances_rle
from src.metrics import iou_matrix
from src.utils import load_config
from src.resources import limit_resources

ANN_COLORS = [(255, 200, 0), (0, 200, 255), (255, 0, 255)]


def draw(img_u8, pred_masks, matched, gt_polys_by_set, out_path, size=1024, zoom=0):
    vis = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    over = vis.copy()
    for m, ok in zip(pred_masks, matched):
        over[m] = (0, 200, 0) if ok else (0, 0, 230)
    vis = cv2.addWeighted(over, 0.35, vis, 0.65, 0)
    for m, ok in zip(pred_masks, matched):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, (0, 255, 0) if ok else (0, 0, 255), 2)
    for c, polys in zip(ANN_COLORS, gt_polys_by_set):
        for p in polys:
            cv2.polylines(vis, [np.round(p).astype(np.int32)], True, c, 1, cv2.LINE_AA)
    cv2.imwrite(out_path, cv2.resize(vis, (size, size), interpolation=cv2.INTER_AREA))
    # Native-resolution crops around the largest annotated filaments (report figures).
    polys = sorted((p for ps in gt_polys_by_set for p in ps), key=lambda p: -np.ptp(p, 0).max())
    done = []
    for p in polys:
        if len(done) >= zoom:
            break
        cx, cy = p.mean(0)
        side = int(min(768, max(256, 1.6 * np.ptp(p, 0).max())))
        if any(abs(cx - x) < s / 2 and abs(cy - y) < s / 2 for x, y, s in done):
            continue
        x0 = int(np.clip(cx - side / 2, 0, vis.shape[1] - side))
        y0 = int(np.clip(cy - side / 2, 0, vis.shape[0] - side))
        crop = cv2.resize(vis[y0:y0 + side, x0:x0 + side], (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_path.replace(".jpg", f"_zoom{len(done)}.jpg"), crop)
        done.append((cx, cy, side))


def main():
    limit_resources(gpu=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/baseline")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--images", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--refiner", action="store_true",
                    help="OOF mode: refine with the fold's own refiner and stack.yaml blend/tau (the submitted outlines)")
    ap.add_argument("--zoom", type=int, default=0, help="also write N native-resolution crops per frame")
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.run, "viz")
    os.makedirs(out_dir, exist_ok=True)

    if args.csv:  # submission on raw frames
        from prepare_data import list_images
        from src.coco_io import obs_id_from_name

        df = pd.read_csv(args.csv, dtype=str)
        df["obs"] = df.filament_id.str.split("_", n=1).str[0]
        for path in list_images(args.images)[: args.n]:
            oid = obs_id_from_name(path)
            raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            masks = [rle_to_mask(counts_to_rle(c)) > 0 for c in df.segmentation_rle[df.obs == oid]]
            draw(raw, masks, [True] * len(masks), [], os.path.join(out_dir, f"test_{oid}.jpg"))
        print(f"-> {out_dir}")
        return

    cfg = load_config(sorted(glob.glob(os.path.join(args.run, "fold*", "config_used.yaml")))[0])
    heads = cfg["model"]["heads"]
    obs = load_observations(cfg["data"]["train_json"])
    disks = load_disks(cfg["data"]["cache_dir"])
    scorer_path = os.path.join(args.run, "scorer.joblib")
    scorer = joblib.load(scorer_path) if os.path.exists(scorer_path) else None
    pp = cfg["postprocess"]
    tuned = os.path.join(args.run, "postprocess_tuned.yaml")
    if os.path.exists(tuned):
        pp = yaml.safe_load(open(tuned))["postprocess"]

    # Prefer multi-annotator frames: they show label noise next to our errors.
    oof = {os.path.basename(p).rsplit("_", 1)[0]: os.path.dirname(p)
           for p in glob.glob(os.path.join(args.run, "fold*", "oof", "*_mask.png"))}
    picks = sorted(oof, key=lambda o: (-obs[o].n_annotators, o))[::7][: args.n]
    for oid in picks:
        probs = load_probs(oof[oid], oid, heads)
        img = load_cached(cfg["data"]["cache_dir"], oid)
        fold = os.path.basename(os.path.dirname(oof[oid]))  # fold<k>
        ref_path = os.path.join(args.run, "refiner", f"{fold}.pt")
        if scorer and args.refiner and os.path.exists(ref_path):
            import torch
            from src.refiner import load_refiners

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            st = yaml.safe_load(open(os.path.join(args.run, "stack.yaml")))
            masks = predict_with_scorer(probs, img, disks.get(oid), scorer, oid[-2:],
                                        refiners=load_refiners([ref_path], dev), device=dev,
                                        blend=st["blend"], tau=st["tau"])
        elif scorer:
            masks = predict_with_scorer(probs, img, disks.get(oid), scorer, oid[-2:])
        else:
            masks = [rle_to_mask(r) > 0 for r in predict_instances_rle(probs, disks.get(oid), pp, 2048)]
        rles = [mask_to_rle(m) for m in masks]
        sets = gt_rles(obs[oid])
        hit = np.zeros(len(masks), bool)
        for g in sets.values():
            m = iou_matrix(g, rles)
            if m.size:
                hit |= (m > 0.5).any(0)
        polys = [[f.polygon for f in fs] for fs in obs[oid].annotations.values()]
        draw(img, masks, hit, polys, os.path.join(out_dir, f"oof_{oid}.jpg"), zoom=args.zoom)
    print(f"{len(picks)} overlays -> {out_dir}")


if __name__ == "__main__":
    main()
