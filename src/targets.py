"""Rasterize per-pixel training targets from polygon annotations.

All targets are averaged over the observation's annotators, giving *soft*
labels in [0, 1]: a pixel two of three experts marked is 0.67. Training on the
consensus (rather than one random annotator) is what the metric rewards,
because each prediction is scored against every annotator separately.

Heads (selectable in the config):
  mask      filled filament polygons (what we are scored on)
  boundary  a band along each polygon outline; lets post-processing split
            neighbouring filaments that touch or nearly touch
  spine     a thick line along each annotated spine; one connected spine per
            filament, so it is a natural instance seed for grouping fragments
"""
from __future__ import annotations

import cv2
import numpy as np

from .coco_io import Observation

_SHIFT = 4  # cv2 sub-pixel precision: coordinates are multiplied by 2**_SHIFT


def _pts(xy: np.ndarray, scale: float, offset: float = 0.0) -> np.ndarray:
    return np.round((xy * scale + offset) * (1 << _SHIFT)).astype(np.int32).reshape(-1, 1, 2)


def rasterize(o: Observation, size: int, heads: list[str], boundary_px: int = 3,
              spine_px: int = 5, offset: float = 0.0) -> np.ndarray:
    """-> float32 (len(heads), size, size) soft targets at `size` x `size`.

    `offset` is added to the scaled vertex coordinates. COCO vertices sit on pixel edges while
    cv2 works in pixel centres, so -0.5 reproduces the organizers' pycocotools masks best
    (data.target_offset; 0.0 = the original targets of all models trained before 10-27)."""
    scale = size / o.width
    out = np.zeros((len(heads), size, size), np.float32)
    n_sets = max(1, len(o.annotations))
    for fils in o.annotations.values():
        layers = {h: np.zeros((size, size), np.uint8) for h in heads}
        for f in fils:
            p = _pts(f.polygon, scale, offset)
            if "mask" in layers:
                cv2.fillPoly(layers["mask"], [p], 1, lineType=cv2.LINE_8, shift=_SHIFT)
            if "boundary" in layers:
                cv2.polylines(layers["boundary"], [p], True, 1, thickness=boundary_px,
                              lineType=cv2.LINE_8, shift=_SHIFT)
            if "spine" in layers and f.spine is not None and len(f.spine) > 1:
                cv2.polylines(layers["spine"], [_pts(f.spine, scale, offset)], False, 1,
                              thickness=spine_px, lineType=cv2.LINE_8, shift=_SHIFT)
        for i, h in enumerate(heads):
            out[i] += layers[h]
    return out / n_sets


def filament_boxes(o: Observation, scale: float = 1.0) -> np.ndarray:
    """(N, 4) x0,y0,x1,y1 boxes of all filaments of all annotators (for crop sampling)."""
    boxes = [np.r_[f.polygon.min(0), f.polygon.max(0)] * scale
             for fils in o.annotations.values() for f in fils]
    return np.asarray(boxes, np.float32).reshape(-1, 4)
