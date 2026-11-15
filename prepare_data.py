"""Preprocess all frames once and build the cross-validation folds.

  python prepare_data.py --config configs/baseline.yaml

Writes to `data.cache_dir`:
  images/<obs_id>.png   limb-flattened, disk-masked uint8 frames (train + test)
  disks.csv             detected disk centre/radius per frame
  folds.csv             train observations with fold, observatory, date, #annotators
  gt_train.csv          ground truth in the organizers' CSV format (for evaluate.py)

Folds are grouped by time: filaments live for days to weeks and the same day is
observed by several GONG sites, so a random split would put near-duplicates of
validation images in training. Observations within the same `group_days`
window share a fold; folds are stratified by observatory.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import date

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from src.coco_io import load_observations, obs_id_from_name, write_gt_csv
from src.solar_disk import preprocess
from src.utils import load_config
from src.resources import limit_resources

IMG_EXT = (".jpg", ".jpeg", ".png")


def _process(args):
    src, dst, clahe, prior = args
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"unreadable image {src}")
    out, disk = preprocess(img, clahe_clip=clahe, use_prior=prior)
    cv2.imwrite(dst, out)
    return obs_id_from_name(src), disk.cx, disk.cy, disk.r, img.shape[0], img.shape[1]


def list_images(d: str) -> list[str]:
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def main():
    limit_resources(gpu=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--workers", type=int, default=None, help="default: the CPU budget (src/resources.py)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = ap.parse_args()
    from src.resources import cpu_budget

    args.workers = args.workers or cpu_budget()
    cfg = load_config(args.config, args.set)
    d = cfg["data"]
    cache = d["cache_dir"]
    os.makedirs(os.path.join(cache, "images"), exist_ok=True)

    obs = load_observations(d["train_json"])
    print(f"train: {len(obs)} observations, "
          f"{sum(o.n_annotators for o in obs.values())} annotation sets, "
          f"{sum(len(f) for o in obs.values() for f in o.annotations.values())} filaments")

    srcs = list_images(d["train_images"])
    if d.get("test_images") and os.path.isdir(d["test_images"]):
        test = list_images(d["test_images"])
        srcs += test
        print(f"test: {len(test)} images")
    missing = set(obs) - {obs_id_from_name(s) for s in srcs}
    if missing:
        print(f"[warn] {len(missing)} annotated observations have no image, e.g. {sorted(missing)[:3]}")

    jobs = [(s, os.path.join(cache, "images", f"{obs_id_from_name(s)}.png"),
             d.get("clahe_clip", 0.0), d.get("disk_prior", True)) for s in srcs]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(tqdm(ex.map(_process, jobs, chunksize=4), total=len(jobs), desc="preprocess"))
    disks = pd.DataFrame(rows, columns=["obs_id", "cx", "cy", "r", "h", "w"])
    disks.to_csv(os.path.join(cache, "disks.csv"), index=False)
    print(f"disk radius: median {disks.r.median():.0f}px, range {disks.r.min():.0f}-{disks.r.max():.0f}")

    # ---- folds
    have = set(disks.obs_id)
    ids = sorted(o for o in obs if o in have)
    meta = pd.DataFrame({
        "obs_id": ids,
        "observatory": [obs[i].observatory for i in ids],
        "date": [obs[i].date for i in ids],
        "n_annotators": [obs[i].n_annotators for i in ids],
        "n_filaments": [int(np.mean([len(f) for f in obs[i].annotations.values()])) for i in ids],
    })
    days = meta.date.map(lambda s: date(int(s[:4]), int(s[4:6]), int(s[6:])).toordinal())
    groups = days // d.get("group_days", 14)
    k = d.get("n_folds", 5)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg["seed"])
    meta["fold"] = -1
    for f, (_, va) in enumerate(sgkf.split(meta, meta.observatory, groups)):
        meta.loc[va, "fold"] = f
    meta.to_csv(os.path.join(cache, "folds.csv"), index=False)
    print(meta.groupby("fold").agg(n=("obs_id", "size"), multi=("n_annotators", lambda s: (s > 1).sum()),
                                   sites=("observatory", "nunique")))

    write_gt_csv(obs, os.path.join(cache, "gt_train.csv"), ids)
    print(f"wrote {cache}/folds.csv, disks.csv, gt_train.csv")


if __name__ == "__main__":
    main()
