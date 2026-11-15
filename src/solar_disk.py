"""Solar-disk geometry and photometric normalization for GONG H-alpha frames.

Why: raw frames differ by observatory, exposure, seeing and clouds, and the
disk is darker towards the limb (limb darkening). Filaments are *relative*
darkenings, so dividing by a smooth radial profile makes "dark = filament"
consistent from disk centre to limb and across sites. Everything off-disk is
zeroed: filaments are only annotated on the disk.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Disk:
    cx: float
    cy: float
    r: float


# GONG H-alpha frames are pre-centred: on the competition data the disk sits at
# (1023.1, 1024.1) +-0.4 px with radius 899.7 +-0.7 px. Otsu-based detection
# fails on ~16% of frames (clouds, gradients), so detection is only trusted
# when it agrees with this prior.
GONG_PRIOR = Disk(1023.1, 1024.1, 899.7)


def detect_disk(img: np.ndarray, prior: Disk | None = GONG_PRIOR, tol: float = 8.0) -> Disk:
    """Detected disk if it is within `tol` px of the prior (or no prior), else the
    prior scaled to the frame size."""
    d = _detect_disk_otsu(img)
    if prior is None:
        return d
    s = img.shape[1] / 2048
    p = Disk(prior.cx * s, prior.cy * s, prior.r * s)
    ok = abs(d.cx - p.cx) < tol * s and abs(d.cy - p.cy) < tol * s and abs(d.r - p.r) < tol * s
    return d if ok else p


def _detect_disk_otsu(img: np.ndarray) -> Disk:
    """Find the solar disk as the largest bright blob (Otsu) and fit a circle
    from its area/centroid (robust to dark filaments and sunspots inside)."""
    blur = cv2.GaussianBlur(img, (0, 0), 5)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        h, w = img.shape
        return Disk(w / 2, h / 2, min(h, w) * 0.45)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (lab == i).astype(np.uint8)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cx, cy), r_enc = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    r_area = np.sqrt(stats[i, cv2.CC_STAT_AREA] / np.pi)
    # The enclosing circle is robust to dark limb features eating into the blob;
    # fall back to the area radius if a bright off-limb artifact inflates it.
    r = r_enc if r_enc < 1.05 * r_area else r_area
    return Disk(float(cx), float(cy), float(r))


def radial_grid(shape, disk: Disk) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.sqrt((xx - disk.cx) ** 2 + (yy - disk.cy) ** 2) / disk.r


def disk_mask(shape, disk: Disk, margin: float = 0.0) -> np.ndarray:
    return radial_grid(shape, disk) <= (1.0 + margin)


def flatten_limb(img: np.ndarray, disk: Disk, n_bins: int = 200,
                 r_max: float = 0.995) -> np.ndarray:
    """Divide by the median radial intensity profile -> flat disk with value ~1."""
    x = img.astype(np.float32)
    rr = radial_grid(img.shape, disk)
    bins = np.clip((rr * n_bins / r_max).astype(np.int32), 0, n_bins - 1)
    # Profile from a 4x-subsampled grid (~260k disk pixels): same medians, ~20x faster.
    xs, rs, bs = x[::4, ::4].ravel(), rr[::4, ::4].ravel(), bins[::4, ::4].ravel()
    on = rs < r_max
    xs, bs = xs[on], bs[on]
    order = np.argsort(bs, kind="stable")
    xs, bs = xs[order], bs[order]
    edges = np.searchsorted(bs, np.arange(n_bins + 1))
    prof = np.full(n_bins, np.nan, np.float32)
    for b in range(n_bins):
        v = xs[edges[b]:edges[b + 1]]
        if v.size > 20:
            prof[b] = np.median(v)
    # The innermost bins hold only a few subsampled pixels; fill them (and any
    # other empty bin) from their neighbours instead of leaving a default value,
    # which used to saturate a white dot at the disk centre.
    ok = ~np.isnan(prof)
    prof = np.interp(np.arange(n_bins), np.flatnonzero(ok), prof[ok]).astype(np.float32) \
        if ok.any() else np.ones(n_bins, np.float32)
    # Smooth the profile so filaments crossing a ring don't imprint on it.
    prof = cv2.GaussianBlur(prof.reshape(1, -1), (1, 0), sigmaX=3).ravel()
    prof = np.maximum(prof, 1.0)
    out = x / prof[bins]
    out[rr >= 1.0] = 0.0
    return out


def preprocess(img: np.ndarray, clahe_clip: float = 0.0, lo: float = 0.4,
               hi: float = 1.3, use_prior: bool = True) -> tuple[np.ndarray, Disk]:
    """uint8 grayscale frame -> (uint8 normalized frame, disk). `lo..hi` maps the
    flattened disk (median 1.0) to 0..255; filaments sit around 0.6-0.9."""
    disk = detect_disk(img, GONG_PRIOR if use_prior else None)
    flat = flatten_limb(img, disk)
    out = np.clip((flat - lo) / (hi - lo), 0, 1)
    out = (out * 255).astype(np.uint8)
    out[~disk_mask(img.shape, disk)] = 0
    if clahe_clip > 0:
        out = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(out)
    return out, disk
