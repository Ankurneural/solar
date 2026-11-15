"""MAGFiLO (COCO-style) annotation loading and COCO-RLE helpers.

Data model
----------
An *observation* is one H-alpha image, named YYYYMMDDHHMMSSII (II = observatory,
e.g. "Bh" Big Bear). Some observations were annotated independently by 2-3
annotators; COCO `images[].id` is then "<annotator>-<observation>" and each of
those is a separate *annotation set*. The official metric scores our single
prediction for an observation against EVERY annotation set of it, so the
targets we should aim for are the annotator consensus, not one annotator.

Each filament is one closed polygon (no holes, never multi-part) plus a spine
polyline. Masks are 2048x2048. Submissions store only the compressed COCO RLE
`counts` string; size is implicitly [2048, 2048].
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from pycocotools import mask as mask_util

IMG_SIZE = 2048
_OBS_RE = re.compile(r"(\d{14}[A-Za-z]{2})")


@dataclass
class Filament:
    polygon: np.ndarray            # (N, 2) float xy, in original 2048 pixel coords
    spine: np.ndarray | None       # (M, 2) float xy or None
    category: int
    area: float


@dataclass
class Observation:
    obs_id: str                    # e.g. "20160920230134Lh"
    file_name: str
    width: int = IMG_SIZE
    height: int = IMG_SIZE
    # annotation-set id ("<annotator>-<obs_id>") -> filaments drawn by that annotator
    annotations: dict[str, list[Filament]] = field(default_factory=dict)

    @property
    def observatory(self) -> str:
        return self.obs_id[-2:]

    @property
    def date(self) -> str:
        return self.obs_id[:8]

    @property
    def n_annotators(self) -> int:
        return len(self.annotations)


def obs_id_from_name(name: str) -> str:
    m = _OBS_RE.search(os.path.basename(name))
    if not m:
        raise ValueError(f"cannot parse observation id from {name!r}")
    return m.group(1)


def _flat_to_xy(flat) -> np.ndarray:
    return np.asarray(flat, dtype=np.float64).reshape(-1, 2)


def load_observations(json_path: str) -> dict[str, Observation]:
    """Parse the competition train JSON into {obs_id: Observation}."""
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    by_set: dict[str, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        by_set[str(a["image_id"])].append(a)

    obs: dict[str, Observation] = {}
    for im in coco["images"]:
        set_id = str(im["id"])
        oid = obs_id_from_name(im["file_name"])
        o = obs.setdefault(oid, Observation(oid, im["file_name"],
                                            im.get("width", IMG_SIZE), im.get("height", IMG_SIZE)))
        fils = []
        for a in by_set.get(set_id, []):
            seg = a["segmentation"]
            polys = seg if isinstance(seg[0], (list, tuple)) else [seg]
            # Documented as a single polygon; keep the largest if ever violated.
            poly = max((_flat_to_xy(p) for p in polys), key=len)
            spine = a.get("spine")
            if spine and isinstance(spine[0], (list, tuple)):
                spine = spine[0]
            fils.append(Filament(poly, _flat_to_xy(spine) if spine else None,
                                 int(a.get("category_id", 0)), float(a.get("area", 0.0))))
        o.annotations[set_id] = fils
    return obs


# --------------------------------------------------------------------------- RLE
def polygon_to_rle(poly_xy: np.ndarray, h: int = IMG_SIZE, w: int = IMG_SIZE,
                   scale: float = 1.0) -> dict:
    """Rasterize a polygon exactly as the organizers do (pycocotools)."""
    flat = (np.asarray(poly_xy, dtype=np.float64) * scale).ravel().tolist()
    rles = mask_util.frPyObjects([flat], h, w)
    return mask_util.merge(rles)


def mask_to_rle(mask: np.ndarray) -> dict:
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))


def rle_counts(rle: dict) -> str:
    c = rle["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else c


def counts_to_rle(counts: str, h: int = IMG_SIZE, w: int = IMG_SIZE) -> dict:
    return {"size": [h, w], "counts": counts.encode("ascii") if isinstance(counts, str) else counts}


def rle_to_mask(rle: dict) -> np.ndarray:
    return mask_util.decode(rle)


def gt_rles(o: Observation, scale: float = 1.0) -> dict[str, list[dict]]:
    """{annotation_set_id: [RLE per filament]} at `scale` x native resolution."""
    h, w = round(o.height * scale), round(o.width * scale)
    return {sid: [polygon_to_rle(f.polygon, h, w, scale) for f in fils]
            for sid, fils in o.annotations.items()}


def write_gt_csv(observations: dict[str, Observation], path: str, obs_ids=None) -> None:
    """Write ground truth in the organizers' GT format (filament_id =
    "<annotator>-<obs>_<k>") so evaluate.py scores it exactly like Kaggle."""
    import pandas as pd

    rows = []
    for oid in obs_ids if obs_ids is not None else sorted(observations):
        for sid, rles in gt_rles(observations[oid]).items():
            rows += [(f"{sid}_{k}", rle_counts(r)) for k, r in enumerate(rles, 1)]
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(path, index=False)
