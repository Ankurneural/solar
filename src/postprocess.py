"""Probability maps -> non-overlapping filament instances.

This stage decides most of the PQ score, because PQ counts objects:
  * a missed filament costs 0.5 (FN); a spurious blob costs 0.5 (FP) whatever its size
  * one filament broken into 2 pieces -> at best 1 TP + 1 FP
  * two filaments merged into one blob -> usually 2 FN + 1 FP
So the knobs below trade fragmentation against merging and recall against
precision; they are tuned on out-of-fold predictions (tune_postprocess.py).

All pixel parameters (close_px, merge_px, grow_px, min_area) are in NATIVE
2048-px units and are rescaled to the resolution of the input probabilities, so
values tuned at 1024 mean the same thing at 2048.

Pipeline:
  1. binarize the mask head at `threshold`, restricted to the solar disk
  2. instance grouping (`grouping`):
       cc     connected components after closing with `close_px`; fragments
              within `merge_px` of each other are grouped into one instance
       spine  seeds = connected components of the spine head; watershed on the
              mask probability assigns pixels to seeds (fragments sharing a
              spine merge; blobs holding two spines split); unseeded blobs
              fall back to `cc`
     `boundary_cut` > 0 additionally removes pixels where the boundary head is
     confident before labelling, to separate touching filaments
  3. per instance: fill holes (GT polygons never have holes), optional `grow_px`
     dilation (annotator polygons are drawn loosely around the dark material)
  4. drop instances with area < `min_area` or mean probability < `min_score`
     (an instance should be kept only if P(match) * E[IoU] > PQ / 2, see notes)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass
class Instance:
    mask: np.ndarray       # bool, full image
    score: float           # mean mask probability inside the instance
    area: int


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _group_cc(binary: np.ndarray, close_px: int, merge_px: int) -> np.ndarray:
    b = binary.astype(np.uint8)
    if close_px > 0:
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, _disk(close_px))
    if merge_px > 0:
        # Label the dilated mask, then give each original pixel its dilated label:
        # nearby fragments share a label without adding the bridge pixels.
        n, lab = cv2.connectedComponents(cv2.dilate(b, _disk(merge_px)), connectivity=8)
        return lab * (b > 0)
    _, lab = cv2.connectedComponents(b, connectivity=8)
    return lab


def _group_spine(binary: np.ndarray, prob: np.ndarray, spine: np.ndarray,
                 p: dict) -> np.ndarray:
    from skimage.segmentation import watershed

    seeds = (spine > p.get("spine_threshold", 0.5)) & binary
    seeds = cv2.morphologyEx(seeds.astype(np.uint8), cv2.MORPH_OPEN, _disk(1))
    n, markers = cv2.connectedComponents(seeds, connectivity=8)
    region = binary.astype(np.uint8)
    if p.get("close_px", 0) > 0:
        region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, _disk(p["close_px"]))
    lab = watershed(-prob, markers, mask=region > 0) if n > 1 else np.zeros_like(markers)
    # Fragments containing no spine seed: label them with the cc rule.
    rest = (region > 0) & (lab == 0)
    if rest.any():
        extra = _group_cc(rest, 0, p.get("merge_px", 0))
        lab = np.where(extra > 0, extra + lab.max(), lab)
    return lab


NATIVE = 2048
_LENGTHS = ("close_px", "merge_px", "grow_px")


def to_working_units(p: dict, size: int) -> dict:
    """Native-px parameters -> pixels at `size` (lengths scale by s, areas by s^2)."""
    s = size / NATIVE
    q = dict(p)
    for k in _LENGTHS:
        if q.get(k):
            q[k] = max(1, int(round(q[k] * s)))
    if q.get("min_area"):
        q["min_area"] = q["min_area"] * s * s
    return q


def instances_from_probs(probs: dict[str, np.ndarray], disk_mask: np.ndarray | None,
                         p: dict) -> list[Instance]:
    """probs: {"mask": HxW float, optional "boundary"/"spine"} -> instances (sorted by score)."""
    prob = probs["mask"].astype(np.float32)
    p = to_working_units(p, prob.shape[0])
    binary = prob > p["threshold"]
    if disk_mask is not None:
        binary &= disk_mask
    if p.get("boundary_cut", 0) > 0 and "boundary" in probs:
        binary &= probs["boundary"] < p["boundary_cut"]

    if p.get("grouping", "cc") == "spine" and "spine" in probs:
        lab = _group_spine(binary, prob, probs["spine"], p)
    else:
        lab = _group_cc(binary, p.get("close_px", 0), p.get("merge_px", 0))

    grow = _disk(p["grow_px"]) if p.get("grow_px", 0) > 0 else None
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        pad = 2 + p.get("grow_px", 0)
        ys = slice(max(0, sl[0].start - pad), min(lab.shape[0], sl[0].stop + pad))
        xs = slice(max(0, sl[1].start - pad), min(lab.shape[1], sl[1].stop + pad))
        m = lab[ys, xs] == i
        area = int(m.sum())
        if area < p.get("min_area", 0):
            continue
        score = float(prob[ys, xs][m].mean())
        if score < p.get("min_score", 0.0):
            continue
        m = ndi.binary_fill_holes(m)
        if grow is not None:
            m = cv2.dilate(m.astype(np.uint8), grow) > 0
        full = np.zeros(lab.shape, bool)
        full[ys, xs] = m
        out.append(Instance(full, score, int(m.sum())))
    out.sort(key=lambda t: -t.score)
    if p.get("max_instances"):
        out = out[: p["max_instances"]]
    return _resolve_overlaps(out, max(1, p.get("min_area", 0)))


def _resolve_overlaps(insts: list[Instance], min_area: int) -> list[Instance]:
    """Growing/closing can make instances overlap. Kaggle rejects overlapping
    masks, so paint in score order: the higher-scoring instance claims contested
    pixels, and leftovers that shrink below `min_area` (or to nothing — empty
    masks must not be submitted) are dropped."""
    if not insts:
        return insts
    taken = np.zeros(insts[0].mask.shape, bool)
    out = []
    for t in insts:
        m = t.mask & ~taken
        area = int(m.sum())
        if area >= min_area:
            taken |= m
            out.append(Instance(m, t.score, area))
    return out
