"""Refiner target round trip vs the organizers' GT rasterization (pycocotools via polygon_to_rle).

  python tools/check_target_geometry.py      # needs runs/w71_b1_r34/refiner/candidates.pkl and runs/w71_b1 OOF maps

For each sampled matched candidate: build the training target (consensus of the matched GT filaments)
in crop space, paste it back to native 2048 and compare (>0.5) with the native consensus GT.
Variants: square (no offset), square with the -0.5 crop-pixel offset fix, rot (principal-axis crop).
Result in docs/EXPERIMENTS.md "Target pixel convention".
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mu

from src.coco_io import load_observations, polygon_to_rle
from src.refiner import NATIVE, crop_box, paste, paste_affine, rot_box
from src.utils import load_config
import train_refiner as tr

R = tr.R
SHIFT = tr._SHIFT
run = "runs/w71_b1_r34"
cfg = load_config(sorted(glob.glob(os.path.join(run, "fold*", "config_used.yaml")))[0])
obs = load_observations(cfg["data"]["train_json"])
t = pd.read_pickle(os.path.join(run, "refiner", "candidates.pkl"))
t = t[(t.y > 0) & (t.match.map(len) > 0)].sample(300, random_state=0).reset_index(drop=True)


def square_target(r, polys, offset):
    cx, cy, side = crop_box(r.bbox)
    k = R / side
    tgt = np.zeros((R, R), np.float32)
    for poly in polys:
        pts = (((poly - [cx - side // 2, cy - side // 2]) * k + offset) * (1 << SHIFT)).round().astype(np.int32)
        layer = np.zeros((R, R), np.uint8)
        cv2.fillPoly(layer, [pts.reshape(-1, 1, 2)], 1, shift=SHIFT)
        tgt += layer
    return tgt, paste(tgt, cx, cy, side)


ds_rot = tr.CropDataset(t, cfg, obs, cfg["model"]["heads"], train=False, geom="rot")
ious = {"square": [], "square_fix": [], "rot": []}
for i in range(len(t)):
    r = t.iloc[i]
    o = obs[r.obs_id]
    n_sets = max(1, len(o.annotations))
    polys = [o.annotations[sid][gi].polygon for sid, gi in r.match.items()]
    gt = np.zeros((NATIVE, NATIVE), np.float32)
    for sid, gi in r.match.items():
        gt += mu.decode(polygon_to_rle(o.annotations[sid][gi].polygon))
    gt = gt / n_sets > 0.5
    if gt.sum() == 0:
        continue
    pasted = {}
    for name, off in (("square", 0.0), ("square_fix", -0.5)):
        tgt, (y0, x0, p) = square_target(r, polys, off)
        pasted[name] = (y0, x0, p / n_sets)
    seed = mu.decode(r.rle)
    pasted["rot"] = paste_affine(ds_rot[i][1].numpy(), rot_box(seed))
    for name, (y0, x0, p) in pasted.items():
        back = np.zeros((NATIVE, NATIVE), bool)
        back[y0:y0 + p.shape[0], x0:x0 + p.shape[1]] = p > 0.5
        ious[name].append((back & gt).sum() / (back | gt).sum())
lines = [f"{k:10s}: median IoU {np.median(v):.4f}, p10 {np.percentile(v, 10):.4f}, mean {np.mean(v):.4f} (n={len(v)})"
         for k, v in ious.items()]
print("\n".join(lines), flush=True)

