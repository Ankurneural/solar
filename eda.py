"""Dataset statistics that drive design decisions, plus the human ceiling.

  python eda.py --config configs/baseline.yaml [--overlays 12]

Prints / saves to <cache_dir>/eda/:
  * observations per observatory, annotators per observation, filaments per set
  * filament area, bbox size and elongation distributions (-> min_area, crop size)
  * gap between neighbouring filaments (-> how far merge_px can go before merging
    distinct filaments)
  * inter-annotator PQ: annotator A scored as if it were a prediction against B.
    This is the realistic ceiling for any model.
  * overlay PNGs of frames with every annotator's polygons
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import Counter

import cv2
import numpy as np
from scipy import ndimage as ndi

from src.coco_io import gt_rles, load_observations
from src.metrics import PQAccumulator
from src.utils import load_config
from src.resources import limit_resources

COLORS = [(255, 64, 64), (64, 255, 64), (64, 160, 255), (255, 220, 0)]


def q(a, ps=(5, 25, 50, 75, 95)):
    a = np.asarray(a, float)
    return {f"p{p}": round(float(np.percentile(a, p)), 1) for p in ps} if a.size else {}


def nearest_gaps(o, scale=0.25) -> list[float]:
    """Min distance (native px) from each filament to any other filament of the same annotator."""
    out = []
    size = round(o.width * scale)
    for fils in o.annotations.values():
        if len(fils) < 2:
            continue
        masks = []
        for f in fils:
            m = np.zeros((size, size), np.uint8)
            cv2.fillPoly(m, [np.round(f.polygon * scale).astype(np.int32)], 1)
            masks.append(m)
        for i, m in enumerate(masks):
            others = np.zeros_like(m)
            for j, mj in enumerate(masks):
                if j != i:
                    others |= mj
            if others.any() and m.any():
                dist = ndi.distance_transform_edt(others == 0)
                out.append(float(dist[m > 0].min()) / scale)
    return out


def main():
    limit_resources(gpu=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--overlays", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    out_dir = os.path.join(d["cache_dir"], "eda")
    os.makedirs(out_dir, exist_ok=True)

    obs = load_observations(d["train_json"])
    fils = [f for o in obs.values() for fs in o.annotations.values() for f in fs]
    areas = [f.area for f in fils]
    wh = np.array([f.polygon.max(0) - f.polygon.min(0) for f in fils])
    elong = wh.max(1) / np.maximum(wh.min(1), 1)
    spine_len = [float(np.linalg.norm(np.diff(f.spine, axis=0), axis=1).sum())
                 for f in fils if f.spine is not None and len(f.spine) > 1]

    stats = {
        "observations": len(obs),
        "annotation_sets": sum(o.n_annotators for o in obs.values()),
        "filaments": len(fils),
        "per_observatory": dict(Counter(o.observatory for o in obs.values()).most_common()),
        "annotators_per_obs": dict(sorted(Counter(o.n_annotators for o in obs.values()).items())),
        "years": dict(sorted(Counter(o.date[:4] for o in obs.values()).items())),
        "filaments_per_set": q([len(fs) for o in obs.values() for fs in o.annotations.values()]),
        "categories": dict(Counter(f.category for f in fils)),
        "area_px": q(areas),
        "bbox_long_side_px": q(wh.max(1)),
        "bbox_short_side_px": q(wh.min(1)),
        "elongation": q(elong),
        "spine_length_px": q(spine_len),
        "missing_spine": sum(f.spine is None for f in fils),
        "frac_foreground_pixels": q([sum(f.area for f in fs) / (o.width * o.height) * 100
                                     for o in obs.values() for fs in o.annotations.values()]),
    }
    gaps = list(itertools.chain.from_iterable(nearest_gaps(o) for o in list(obs.values())[:300]))
    stats["nearest_neighbour_gap_px"] = q(gaps)

    # Human ceiling: every ordered pair of annotators of the same observation.
    acc = PQAccumulator()
    for o in obs.values():
        if o.n_annotators < 2:
            continue
        sets = list(gt_rles(o).values())
        for a, b in itertools.permutations(range(len(sets)), 2):
            acc.update(sets[b], sets[a])
    stats["inter_annotator"] = {k: round(v, 4) if isinstance(v, float) else v
                                for k, v in acc.compute().items()}

    print(json.dumps(stats, indent=1))
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)

    # Overlays: multi-annotator frames first, they show the label noise best.
    picks = sorted(obs.values(), key=lambda o: -o.n_annotators)[: args.overlays]
    for o in picks:
        p = os.path.join(d["cache_dir"], "images", f"{o.obs_id}.png")
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for c, fs in zip(COLORS, o.annotations.values()):
            for f in fs:
                cv2.polylines(vis, [np.round(f.polygon).astype(np.int32)], True, c, 2)
                if f.spine is not None:
                    cv2.polylines(vis, [np.round(f.spine).astype(np.int32)], False, c, 1)
        cv2.imwrite(os.path.join(out_dir, f"overlay_{o.obs_id}.jpg"),
                    cv2.resize(vis, (1024, 1024), interpolation=cv2.INTER_AREA))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
