"""Figure 1 of the report: the pipeline on one real out-of-fold frame, plus the stage schematic.

  python tools/make_figure_pipeline.py --run runs/w11_all_768b --obs 20110429135554Bh

Panels (native-resolution crop around the frame's largest annotated filament):
  a) pre-processed full disk with the crop box   b) crop
  c) stage-1 ensemble probability (OOF)          d) candidate tree: components at three of the 10 threshold levels
  e) final output (GBM + rotated refiner, the fold's own models) with the three annotators' outlines
Writes runs/figures/fig1_pipeline.{pdf,png}. Runs on the CPU (the GPU stays free for training).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from src.candidates import predict_with_scorer
from src.coco_io import load_observations
from src.dataset import load_cached
from src.evaluation import load_disks, load_probs
from src.refiner import load_refiners
from src.resources import limit_resources
from src.utils import load_config

ANN = ["#ffc800", "#00c8ff", "#ff00ff"]


def main():
    limit_resources(gpu=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/w11_all_768b")
    ap.add_argument("--obs", default="20110429135554Bh")
    ap.add_argument("--out", default="runs/figures/fig1_pipeline")
    args = ap.parse_args()
    cfg = load_config(sorted(glob.glob(os.path.join(args.run, "fold*", "config_used.yaml")))[0])
    obs = load_observations(cfg["data"]["train_json"])[args.obs]
    oof_dir = os.path.dirname(glob.glob(os.path.join(args.run, "fold*", "oof", f"{args.obs}_mask.png"))[0])
    fold = os.path.basename(os.path.dirname(oof_dir))
    probs = load_probs(oof_dir, args.obs, cfg["model"]["heads"])
    img = load_cached(cfg["data"]["cache_dir"], args.obs)
    disk = load_disks(cfg["data"]["cache_dir"]).get(args.obs)
    scorer = joblib.load(os.path.join(args.run, "scorer.joblib"))
    st = yaml.safe_load(open(os.path.join(args.run, "stack.yaml")))
    refiners = load_refiners([os.path.join(args.run, "refiner", f"{fold}.pt")], "cpu")
    masks = predict_with_scorer(probs, img, disk, scorer, args.obs[-2:], refiners=refiners, device="cpu",
                                blend=st["blend"], tau=st["tau"])

    polys = [[f.polygon for f in fs] for fs in obs.annotations.values()]
    big = max((p for ps in polys for p in ps), key=lambda p: np.ptp(p, 0).max())
    side = int(min(768, max(320, 1.8 * np.ptp(big, 0).max())))
    cx, cy = big.mean(0)
    x0 = int(np.clip(cx - side / 2, 0, 2048 - side))
    y0 = int(np.clip(cy - side / 2, 0, 2048 - side))
    sl = (slice(y0, y0 + side), slice(x0, x0 + side))
    prob = cv2.resize(probs["mask"].astype(np.float32) / (255 if probs["mask"].dtype == np.uint8 else 1),
                      (2048, 2048), interpolation=cv2.INTER_LINEAR)

    fig = plt.figure(figsize=(14, 4.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 0.3], hspace=0.12, wspace=0.04)
    ax = [fig.add_subplot(gs[0, i]) for i in range(5)]
    ax[0].imshow(img, cmap="gray")
    ax[0].add_patch(Rectangle((x0, y0), side, side, fill=False, ec="red", lw=1.5))
    ax[1].imshow(img[sl], cmap="gray")
    ax[2].imshow(prob[sl], cmap="magma", vmin=0, vmax=1)
    ax[3].imshow(img[sl], cmap="gray")
    for lvl, col in zip((0.3, 0.5, 0.7), ("#4da6ff", "#ffd24d", "#ff4d4d")):
        cs, _ = cv2.findContours((prob[sl] > lvl).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            ax[3].plot(c[:, 0, 0], c[:, 0, 1], color=col, lw=1)
    ax[4].imshow(img[sl], cmap="gray")
    over = np.zeros(img[sl].shape + (4,))
    for m in masks:
        over[m[sl]] = (0.1, 0.9, 0.1, 0.45)
    ax[4].imshow(over)
    for col, ps in zip(ANN, polys):
        for p in ps:
            q = np.vstack([p, p[:1]]) - [x0, y0]
            ax[4].plot(q[:, 0], q[:, 1], color=col, lw=1)
    titles = ["a) pre-processed disk", "b) native-resolution crop", "c) stage-1 probability",
              "d) candidate tree (t=0.3/0.5/0.7)", "e) output vs 3 annotators"]
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=9.5)
        a.set_xticks([]), a.set_yticks([])
    for a in ax[1:]:
        a.set_xlim(0, side), a.set_ylim(side, 0)

    sch = fig.add_subplot(gs[1, :])
    sch.set_xlim(0, 100), sch.set_ylim(0, 10), sch.axis("off")
    boxes = [
        (1, "H$\\alpha$ frame 2048$^2$\ndisk fit, limb\ncorrection"),
        (21, "Stage 1: U-Net ensemble\nConvNeXt crops 512/768\n+ full disk 1024"),
        (41, "Candidate tree\n10 threshold levels\n~100 candidates / frame"),
        (61, "Scoring\nGBM (42 features)\n+ rotated refiner"),
        (81, "Selection: tree DP\n$q>\\tau$, refined\noutlines, no overlap"),
    ]
    for x, txt in boxes:
        sch.add_patch(FancyBboxPatch((x, 1.2), 17, 7.6, boxstyle="round,pad=0.3", fc="#eef3fb", ec="#36598c"))
        sch.text(x + 8.5, 5, txt, ha="center", va="center", fontsize=8.5)
    for x, _ in boxes[:-1]:
        sch.add_patch(FancyArrowPatch((x + 17.4, 5), (x + 19.6, 5), arrowstyle="-|>", mutation_scale=14, color="#36598c"))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", dpi=130, bbox_inches="tight")
    print(f"{len(masks)} predicted filaments; -> {args.out}.pdf/.png")


if __name__ == "__main__":
    main()
