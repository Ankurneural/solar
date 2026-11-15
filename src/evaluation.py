"""Glue between probability maps, post-processing and the PQ metric, plus the
submission writer. Shared by train.py (validation), tune_postprocess.py,
infer.py and evaluate.py so every stage scores things the same way."""
from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd

from .coco_io import IMG_SIZE, Observation, counts_to_rle, gt_rles, mask_to_rle, rle_counts
from .metrics import PQAccumulator
from .postprocess import instances_from_probs
from .solar_disk import Disk, disk_mask


# ----------------------------------------------------------------- disk geometry
def load_disks(cache_dir: str) -> dict[str, Disk]:
    df = pd.read_csv(os.path.join(cache_dir, "disks.csv"))
    return {r.obs_id: Disk(r.cx, r.cy, r.r) for r in df.itertuples()}


def disk_mask_at(disk: Disk | None, size: int, margin: float) -> np.ndarray | None:
    if disk is None:
        return None
    s = size / IMG_SIZE
    return disk_mask((size, size), Disk(disk.cx * s, disk.cy * s, disk.r * s), margin)


# ----------------------------------------------------------------- prob caching
def save_probs(out_dir: str, obs_id: str, probs: dict[str, np.ndarray]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for h, p in probs.items():
        cv2.imwrite(os.path.join(out_dir, f"{obs_id}_{h}.png"),
                    np.clip(p * 255 + 0.5, 0, 255).astype(np.uint8))


def load_probs(out_dir: str, obs_id: str, heads: list[str],
               as_uint8: bool = False) -> dict[str, np.ndarray]:
    """as_uint8 keeps maps 4x smaller in RAM (the whole OOF set for tuning);
    resize_probs converts them to float on use."""
    out = {h: cv2.imread(os.path.join(out_dir, f"{obs_id}_{h}.png"), cv2.IMREAD_GRAYSCALE)
           for h in heads}
    return out if as_uint8 else {h: v.astype(np.float32) / 255.0 for h, v in out.items()}


# ----------------------------------------------------------------- scoring
def member_oof_dirs(run: str) -> list[str]:
    """Per-member OOF run dirs for member-disagreement scorer features, if the
    ensemble opts in with `member_features: true` (off by default: on ENS2 they
    scored 0.4448 vs 0.4457 without, i.e. no gain; see docs/EXPERIMENTS.md)."""
    import yaml

    p = os.path.join(run, "ensemble.yaml")
    if not os.path.exists(p):
        return []
    ens = yaml.safe_load(open(p, encoding="utf-8"))
    return (ens.get("oof_dirs") or []) if ens.get("member_features") else []


def add_member_maps(probs: dict, member_dirs: list[str], fold: str, oid: str, size: int) -> dict:
    """probs + {"member<i>": member i's OOF mask map at `size`} (uint8 kept as uint8)."""
    for i, d in enumerate(member_dirs):
        m = cv2.imread(os.path.join(d, fold, "oof", f"{oid}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"{d}/{fold}/oof/{oid}_mask.png")
        probs[f"member{i}"] = m if m.shape[0] == size else cv2.resize(m, (size, size),
                                                                   interpolation=cv2.INTER_AREA)
    return probs


def resize_probs(probs: dict[str, np.ndarray], size: int) -> dict[str, np.ndarray]:
    """Resize maps to size x size (uint8 maps become float 0..1). Downsampling uses
    area averaging, the same as make_ensemble.py, so OOF and test maps match."""
    out = {}
    for k, v in probs.items():
        v = v.astype(np.float32) / 255.0 if v.dtype == np.uint8 else v
        if v.shape[0] != size:
            interp = cv2.INTER_AREA if v.shape[0] > size else cv2.INTER_LINEAR
            v = cv2.resize(v, (size, size), interpolation=interp)
        out[k] = v
    return out


def predict_instances_rle(probs, disk, pp: dict, size: int) -> list[dict]:
    probs = resize_probs(probs, size)
    insts = instances_from_probs(probs, disk_mask_at(disk, size, pp.get("disk_margin", 0.01)), pp)
    return [mask_to_rle(t.mask) for t in insts]


def score_obs(observations: dict[str, Observation], probs_by_obs: dict, disks: dict,
              pp: dict, eval_size: int) -> dict:
    """PQ of post-processed predictions at `eval_size` (2048 = exact Kaggle scoring;
    1024 is ~4x faster and close enough for model selection / tuning)."""
    acc = PQAccumulator()
    scale = eval_size / IMG_SIZE
    for oid, probs in probs_by_obs.items():
        o = observations[oid]
        pred = predict_instances_rle(probs, disks.get(oid), pp, eval_size)
        for g in gt_rles(o, scale).values():
            acc.update(g, pred)
    return acc.compute()


# ----------------------------------------------------------------- submission
def submission_rows(obs_id: str, rles: list[dict]) -> list[tuple[str, str]]:
    return [(f"{obs_id}_{k}", rle_counts(r)) for k, r in enumerate(rles, 1)]


def write_submission(rows: list[tuple[str, str]], path: str) -> None:
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(path, index=False)


def read_rle_csv(path: str) -> dict[str, list[dict]]:
    """CSV (filament_id, segmentation_rle) -> {key before first '_': [RLE]}, the
    same grouping the organizers' notebook uses."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out: dict[str, list[dict]] = {}
    for fid, counts in zip(df["filament_id"], df["segmentation_rle"]):
        if counts:
            out.setdefault(fid.split("_", 1)[0], []).append(counts_to_rle(counts))
    return out
