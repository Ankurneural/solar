"""Our fast PQ must equal the organizers' notebook implementation (dense torch
matmul over decoded masks), including the RLE round-trip used in submissions.

  python -m pytest tests -q
"""
from __future__ import annotations

import numpy as np
import torch

from src.coco_io import counts_to_rle, mask_to_rle, polygon_to_rle, rle_counts, rle_to_mask
from src.metrics import PQAccumulator


def notebook_pq(sets):
    """Verbatim logic of get_overlap_matrices + get_pq_score from the notebook."""
    tp, fp, fn = [], 0, 0
    for gt, pred in sets:
        if len(gt) == 0:
            fp += len(pred)
            continue
        if len(pred) == 0:
            fn += len(gt)
            continue
        g = torch.from_numpy(np.stack(gt).reshape(len(gt), -1).astype(np.float32))
        p = torch.from_numpy(np.stack(pred).reshape(len(pred), -1).astype(np.float32))
        inter = g @ p.t()
        union = g.sum(1).view(-1, 1) + p.sum(1).view(1, -1) - inter
        iou = torch.where(union == 0, torch.tensor(0.), inter / union)
        hit = iou > 0.5
        tp += iou[hit].tolist()
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
    den = len(tp) + 0.5 * fp + 0.5 * fn
    return sum(tp) / den if den else 0.0


def _blob(rng, h=128):
    m = np.zeros((h, h), np.uint8)
    y, x = rng.integers(0, h - 30, 2)
    m[y:y + rng.integers(5, 30), x:x + rng.integers(5, 30)] = 1
    return m


def test_pq_matches_notebook():
    rng = np.random.default_rng(0)
    sets = []
    for _ in range(30):
        gt = [_blob(rng) for _ in range(rng.integers(0, 4))]
        pred = [g if rng.random() < 0.5 else _blob(rng) for g in gt] + \
               [_blob(rng) for _ in range(rng.integers(0, 3))]
        sets.append((gt, pred))
    acc = PQAccumulator()
    for gt, pred in sets:
        acc.update([mask_to_rle(m) for m in gt], [mask_to_rle(m) for m in pred])
    assert abs(acc.compute()["pq"] - notebook_pq(sets)) < 1e-9


def test_rle_counts_roundtrip():
    rng = np.random.default_rng(1)
    m = _blob(rng)
    counts = rle_counts(mask_to_rle(m))
    assert isinstance(counts, str)
    assert np.array_equal(rle_to_mask(counts_to_rle(counts, 128, 128)), m)


def test_polygon_rasterization():
    sq = np.array([[10, 10], [40, 10], [40, 30], [10, 30], [10, 10]], float)
    m = rle_to_mask(polygon_to_rle(sq, 64, 64))
    assert abs(int(m.sum()) - 30 * 20) <= 60
