"""Sliding-window tiling and Hann-window blending (from the crack/scratch
pipeline). Used when predicting at native 2048 resolution with a model trained
on crops."""
from __future__ import annotations

import numpy as np


def tile_starts(length: int, tile: int, stride: int) -> list[int]:
    """Start coordinates covering [0, length) with the last tile flush to the edge."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def iter_tiles(h: int, w: int, tile: int, overlap: int):
    """Yield (y0, x0, y1, x1) tile boxes covering an (h, w) image."""
    stride = max(1, tile - overlap)
    for y0 in tile_starts(h, tile, stride):
        for x0 in tile_starts(w, tile, stride):
            yield y0, x0, min(y0 + tile, h), min(x0 + tile, w)


def hann2d(tile: int) -> np.ndarray:
    """2D Hann window (with a small floor so edge tiles still contribute)."""
    w = np.hanning(tile + 2)[1:-1]
    return np.clip(np.outer(w, w).astype(np.float32), 1e-3, None)
