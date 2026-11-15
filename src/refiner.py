"""Stage-2 crop refiner: one candidate at a time, at native resolution.

The first-stage model sees the whole disk. The refiner zooms into one candidate
filament and answers two questions the metric cares about:
  * extent  -> a refined mask of *this* filament (12% of GT filaments are found
               with IoU 0.3-0.5, i.e. just below the 0.5 hit threshold)
  * worth   -> a scalar score: expected IoU*[IoU>0.5] over annotators, the same
               target as the GBM scorer but predicted from pixels

Input channels (square crop around the candidate, resized to `size`):
  0 preprocessed H-alpha image   1 stage-1 mask probability   2 candidate (seed) mask
Target mask: for every annotator, that annotator's GT filament overlapping the
seed most (IoU > 0.1); averaged over annotators, so it is the consensus of
*this* filament, and 0 where annotators didn't draw it.

Crop geometry (`geom`, stored in each checkpoint):
  square  axis-aligned square window around the bbox, resized to 256x256
  rot     window aligned with the seed's principal axis (long axis horizontal),
          ROT_W x ROT_H; long thin filaments get ~2x the pixels of a square crop.
          A "box" is then a 2x3 affine matrix native -> crop pixels.
"""
from __future__ import annotations

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

NATIVE = 2048
ROT_W, ROT_H = 384, 160


def crop_box(bbox_native: tuple[float, float, float, float], pad_frac: float = 0.5,
             min_side: int = 96) -> tuple[int, int, int]:
    """(x0, y0, x1, y1) native -> square (cx, cy, side) window around it."""
    x0, y0, x1, y1 = bbox_native
    side = max(x1 - x0, y1 - y0) * (1 + pad_frac) + 32
    side = int(max(min_side, side))
    return int((x0 + x1) / 2), int((y0 + y1) / 2), side


def extract(arr: np.ndarray, cx: int, cy: int, side: int, size: int,
            interp=cv2.INTER_AREA) -> np.ndarray:
    """Square crop centred at (cx, cy) (zero-padded off-frame), resized to size."""
    h, w = arr.shape
    x0, y0 = cx - side // 2, cy - side // 2
    out = np.zeros((side, side), arr.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + side), min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (size, size), interpolation=interp)


def paste(crop_prob: np.ndarray, cx: int, cy: int, side: int, size: int = NATIVE) -> tuple:
    """Refined crop probability -> (y0, x0, prob at native scale) clipped to the frame."""
    p = cv2.resize(crop_prob, (side, side), interpolation=cv2.INTER_LINEAR)
    x0, y0 = cx - side // 2, cy - side // 2
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(size, x0 + side), min(size, y0 + side)
    return sy0, sx0, p[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]


def rot_box(seed: np.ndarray, pad_frac: float = 0.5, max_zoom: float = 256 / 96) -> np.ndarray:
    """Seed mask (any resolution, square frame) -> 2x3 affine native -> ROT_W x ROT_H crop
    that puts the seed's principal axis horizontal and centres its extent."""
    s = NATIVE / seed.shape[0]
    ys, xs = np.nonzero(seed)
    pts = np.column_stack([xs, ys]).astype(np.float64) * s + (s - 1) / 2
    if len(pts) == 0:
        pts = np.array([[NATIVE / 2, NATIVE / 2]])
    c = pts.mean(0)
    d = pts - c
    cov = d.T @ d / max(1, len(d))
    th = 0.5 * np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1])
    u, v = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    pu, pv = d @ u, d @ v
    c = c + u * (pu.max() + pu.min()) / 2 + v * (pv.max() + pv.min()) / 2
    length, width = pu.max() - pu.min() + s, pv.max() - pv.min() + s
    k = min(ROT_W / (length * (1 + pad_frac) + 32), ROT_H / (width * (1 + pad_frac) + 32), max_zoom)
    A = np.array([[u[0], u[1], 0.0], [v[0], v[1], 0.0]]) * k
    A[:, 2] = np.array([ROT_W / 2, ROT_H / 2]) - A[:, :2] @ c
    return A


def jitter_affine(A: np.ndarray, rng=np.random) -> np.ndarray:
    """Training jitter of a rot box: shift +-10%, zoom 0.87-1.11, rotate +-10 deg about the crop centre."""
    J = cv2.getRotationMatrix2D((ROT_W / 2, ROT_H / 2), rng.uniform(-10, 10), 1 / rng.uniform(0.9, 1.15))
    J[:, 2] += [rng.uniform(-0.1, 0.1) * ROT_W, rng.uniform(-0.1, 0.1) * ROT_H]
    return (np.vstack([J, [0, 0, 1]]) @ np.vstack([A, [0, 0, 1]]))[:2]


def extract_affine(arr: np.ndarray, A: np.ndarray, interp=cv2.INTER_LINEAR) -> np.ndarray:
    """Warp a square full-frame array (any resolution) into the ROT_W x ROT_H crop of box A."""
    s = arr.shape[0] / NATIVE
    Aa = A.copy()
    Aa[:, :2] /= s
    Aa[:, 2] += A[:, :2] @ np.array([(1 - s) / (2 * s)] * 2)  # pixel-centre offset between resolutions
    return cv2.warpAffine(arr, Aa, (ROT_W, ROT_H), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def paste_affine(crop_prob: np.ndarray, A: np.ndarray, size: int = NATIVE) -> tuple:
    """Inverse of extract_affine for a crop probability -> (y0, x0, prob in the covered native ROI)."""
    h, w = crop_prob.shape
    Ai = cv2.invertAffineTransform(A)
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64) @ Ai.T
    x0, y0 = np.clip(np.floor(corners.min(0)).astype(int), 0, size)
    x1, y1 = np.clip(np.ceil(corners.max(0)).astype(int) + 1, 0, size)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, np.zeros((0, 0), np.float32)
    Ar = A.copy()
    Ar[:, 2] += A[:, :2] @ np.array([x0, y0], np.float64)
    p = cv2.warpAffine(crop_prob, Ar, (int(x1 - x0), int(y1 - y0)),
                       flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
    return int(y0), int(x0), p


def make_box(geom: str, bbox_native: tuple, seed: np.ndarray):
    return crop_box(bbox_native) if geom == "square" else rot_box(seed)


def box_inputs(img_u8: np.ndarray, prob: np.ndarray, seed: np.ndarray, box, size: int = 256) -> np.ndarray:
    if isinstance(box, tuple):
        return crop_inputs(img_u8, prob, seed, *box, size)
    p = prob.astype(np.float32) / 255 if prob.dtype == np.uint8 else prob.astype(np.float32)
    return make_input(extract_affine(img_u8, box),
                      extract_affine(p, box),
                      extract_affine(seed.astype(np.float32), box) > 0.5)


def box_paste(crop_prob: np.ndarray, box) -> tuple:
    return paste(crop_prob, *box) if isinstance(box, tuple) else paste_affine(crop_prob, box)


def build_refiner(encoder: str = "resnet18", pretrained: bool = True) -> nn.Module:
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1, aux_params={"classes": 1, "dropout": 0.2})


def make_input(img_c: np.ndarray, prob_c: np.ndarray, seed_c: np.ndarray) -> np.ndarray:
    return np.stack([(img_c.astype(np.float32) / 255 - 0.5) / 0.25,
                     prob_c.astype(np.float32) * 2 - 1,
                     seed_c.astype(np.float32) * 2 - 1])


_throttle = None


@torch.no_grad()
def refine_batch(model, inputs: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """inputs (N, 3, S, S) -> (mask probs (N, S, S), scores (N,)) with flip TTA."""
    global _throttle
    if _throttle is None:
        from .resources import GpuThrottle

        _throttle = GpuThrottle()
    _throttle.begin()
    x = torch.from_numpy(inputs).to(device)
    from .inference import amp_forward

    m1, s1 = amp_forward(model, x, device == "cuda")
    m2, s2 = amp_forward(model, torch.flip(x, (3,)), device == "cuda")
    m = (torch.sigmoid(m1.float()) + torch.flip(torch.sigmoid(m2.float()), (3,))) / 2
    s = (torch.sigmoid(s1.float()) + torch.sigmoid(s2.float())) / 2
    out = m[:, 0].cpu().numpy(), s[:, 0].cpu().numpy()
    _throttle.step(sync=False)  # .cpu() above already synchronised
    return out


def crop_inputs(img_u8: np.ndarray, prob: np.ndarray, seed: np.ndarray, cx: int, cy: int,
                side: int, size: int) -> np.ndarray:
    """(3, size, size) refiner input for one candidate. `img_u8` is the native
    preprocessed frame; `prob` (float 0..1 or uint8) and `seed` (bool/float) may
    be at any resolution: coordinates are scaled to each array's size."""
    ps, ss = prob.shape[0] / NATIVE, seed.shape[0] / NATIVE
    p = prob.astype(np.float32) / 255 if prob.dtype == np.uint8 else prob.astype(np.float32)
    return make_input(
        extract(img_u8, cx, cy, side, size),
        extract(p, int(cx * ps), int(cy * ps), int(side * ps), size, cv2.INTER_LINEAR),
        extract(seed.astype(np.float32), int(cx * ss), int(cy * ss), int(side * ss), size,
                cv2.INTER_LINEAR) > 0.5)


def largest_component(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.uint8)
    n, lab = cv2.connectedComponents(m)
    if n > 2:
        m = lab == (1 + np.argmax(np.bincount(lab.ravel())[1:]))
    return m > 0


def refined_native(crop_prob: np.ndarray, box, thr: float = 0.5) -> np.ndarray:
    """Refined crop probability -> native bool mask (largest component only)."""
    y0, x0, p = box_paste(crop_prob, box)
    m = largest_component(p > thr)
    full = np.zeros((NATIVE, NATIVE), bool)
    full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
    return full


def load_refiners(paths: list[str], device: str) -> list[nn.Module]:
    models = []
    for p in paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        m = build_refiner(ck.get("encoder", "resnet18"), pretrained=False)
        m.load_state_dict(ck["model"])
        m.geom = ck.get("geom", "square")
        models.append(m.to(device).eval())
    if len({m.geom for m in models}) > 1:
        raise ValueError("refiners with different crop geometries cannot be ensembled")
    return models


def run_refiners(models, img_u8: np.ndarray, prob: np.ndarray, seeds: list[np.ndarray],
                 bboxes: list[tuple], device: str, size: int = 256, batch: int = 64):
    """Ensemble of refiners on candidate crops -> (crop probs (N, h, w), scores (N,), boxes)."""
    geom = getattr(models[0], "geom", "square")
    boxes = [make_box(geom, b, s) for b, s in zip(bboxes, seeds)]
    probs, scores = [], []
    for i in range(0, len(seeds), batch):
        x = np.stack([box_inputs(img_u8, prob, s, bx, size)
                      for s, bx in zip(seeds[i:i + batch], boxes[i:i + batch])])
        n = len(x)
        # Pad to a fixed batch: with cudnn.benchmark every new input shape re-runs
        # kernel autotuning, and the candidate count differs on every frame
        # (this cost ~25 s/frame before padding).
        if n < batch:
            x = np.concatenate([x, np.zeros((batch - n,) + x.shape[1:], x.dtype)])
        pm, ps = None, None
        for m in models:
            p, s = refine_batch(m, x, device)
            pm, ps = (p, s) if pm is None else (pm + p, ps + s)
        probs.append(pm[:n] / len(models))
        scores.append(ps[:n] / len(models))
    if not probs:
        h, w = (size, size) if geom == "square" else (ROT_H, ROT_W)
        return np.zeros((0, h, w), np.float32), np.zeros(0), boxes
    return np.concatenate(probs), np.concatenate(scores), boxes


def refine_masks(models, img_u8: np.ndarray, prob: np.ndarray, seeds: list[np.ndarray],
                 bboxes: list[tuple], device: str, size: int = 256, batch: int = 64) -> list[np.ndarray]:
    """Refine candidate masks with an ensemble of refiners -> native bool masks."""
    crops, _, boxes = run_refiners(models, img_u8, prob, seeds, bboxes, device, size, batch)
    return [refined_native(p, bx) for p, bx in zip(crops, boxes)]
