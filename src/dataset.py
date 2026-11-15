"""Training dataset over preprocessed full-disk frames.

Two sampling modes (config `data.mode`):
  full  the whole disk resized to `input_size` (e.g. 1024). Gives the network
        global context, which matters for deciding where one filament ends and
        the next begins. Cheapest; good baseline.
  crop  `crop_size` windows from the frame scaled by `scale` (1.0 = native
        2048). Preserves thin barbs; crops are centred on a random filament with
        probability `pos_frac`, otherwise anywhere on the disk (hard negatives:
        sunspots, plage, cloud edges).
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .coco_io import Observation
from .targets import filament_boxes, rasterize


def load_cached(cache_dir: str, obs_id: str) -> np.ndarray:
    p = os.path.join(cache_dir, "images", f"{obs_id}.png")
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img


def normalize(img_u8: np.ndarray) -> np.ndarray:
    """uint8 -> float32 roughly zero-mean/unit-var on the disk."""
    return (img_u8.astype(np.float32) / 255.0 - 0.5) / 0.25


class FilamentDataset(Dataset):
    def __init__(self, observations: list[Observation], cfg: dict, transforms=None,
                 train: bool = True):
        self.obs = observations
        self.cfg = cfg
        self.d = cfg["data"]
        self.heads = cfg["model"]["heads"]
        self.tf = transforms
        self.train = train
        self.repeats = self.d.get("repeats", 1) if train else 1

    def __len__(self) -> int:
        return len(self.obs) * self.repeats

    def _crop_origin(self, o: Observation, size: int, crop: int, scale: float,
                     rng: np.random.Generator) -> tuple[int, int]:
        hi = max(0, size - crop)
        boxes = filament_boxes(o, scale)
        if self.train and len(boxes) and rng.random() < self.d.get("pos_frac", 0.7):
            x0, y0, x1, y1 = boxes[rng.integers(len(boxes))]
            cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
            jitter = crop * 0.25
            cx += rng.uniform(-jitter, jitter)
            cy += rng.uniform(-jitter, jitter)
        elif self.train:
            # Uniform over the disk (rejection-free: disk ~ inscribed in frame).
            r = np.sqrt(rng.random()) * size * 0.45
            t = rng.uniform(0, 2 * np.pi)
            cx, cy = size / 2 + r * np.cos(t), size / 2 + r * np.sin(t)
        else:
            cx = cy = size / 2
        return (int(np.clip(cy - crop / 2, 0, hi)), int(np.clip(cx - crop / 2, 0, hi)))

    def __getitem__(self, idx: int):
        o = self.obs[idx % len(self.obs)]
        rng = np.random.default_rng() if self.train else np.random.default_rng(idx)
        img = load_cached(self.d["cache_dir"], o.obs_id)

        if self.d["mode"] == "full":
            size = self.d["input_size"]
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            tgt = rasterize(o, size, self.heads, self.d.get("boundary_px", 3),
                            self.d.get("spine_px", 5), self.d.get("target_offset", 0.0))
        else:
            scale, crop = self.d.get("scale", 1.0), self.d["crop_size"]
            size = round(o.width * scale)
            if size != img.shape[0]:
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            tgt = rasterize(o, size, self.heads, self.d.get("boundary_px", 3),
                            self.d.get("spine_px", 5), self.d.get("target_offset", 0.0))
            y0, x0 = self._crop_origin(o, size, crop, scale, rng)
            img = img[y0:y0 + crop, x0:x0 + crop]
            tgt = tgt[:, y0:y0 + crop, x0:x0 + crop]

        if self.tf is not None:
            res = self.tf(image=img, masks=list(tgt))
            img, tgt = res["image"], np.stack(res["masks"])

        x = torch.from_numpy(normalize(img))[None]
        y = torch.from_numpy(np.ascontiguousarray(tgt, dtype=np.float32))
        return x, y
