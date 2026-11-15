"""Losses for soft (annotator-averaged) multi-head targets.

  bce      pixel-wise; handles soft targets natively
  dice     soft Dice per image; counteracts the ~1-2% foreground imbalance and
           tracks the IoU we are scored on
  cldice   soft-skeleton clDice (Shit et al., CVPR 2021): rewards topological
           continuity, i.e. not breaking a long thin filament into pieces,
           which the metric punishes (a split GT is often 1 TP + 1 FP or worse).
           Off by default: public write-ups report clDice/Lovasz making
           probabilities less calibrated, and selection relies on calibration.
  bicc     bidirectional connected-component loss (opt-in): one equal-weight term per
           consensus component and per false-positive blob, mirroring PQ's RQ, so
           small filaments and spurious blobs count as much as large ones.

Per-head weights come from `loss.heads.<head>`; e.g. the auxiliary boundary and
spine heads use BCE+Dice with a lower weight than the mask head.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi


def soft_dice_loss(prob: torch.Tensor, tgt: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    dims = (1, 2)
    inter = (prob * tgt).sum(dims)
    den = prob.sum(dims) + tgt.sum(dims)
    return (1 - (2 * inter + eps) / (den + eps)).mean()


def _soft_erode(x):
    return -F.max_pool2d(-x, 3, 1, 1)


def _soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def soft_skeleton(x: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Differentiable morphological skeleton of a (B, 1, H, W) probability map."""
    def _open(t):
        return _soft_dilate(_soft_erode(t))
    skel = F.relu(x - _open(x))
    for _ in range(iters):
        x = _soft_erode(x)
        delta = F.relu(x - _open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(prob: torch.Tensor, tgt: torch.Tensor, iters: int = 10,
                     eps: float = 1.0) -> torch.Tensor:
    p, t = prob[:, None], tgt[:, None]
    sp, st = soft_skeleton(p, iters), soft_skeleton(t, iters)
    tprec = ((sp * t).sum((1, 2, 3)) + eps) / (sp.sum((1, 2, 3)) + eps)
    tsens = ((st * p).sum((1, 2, 3)) + eps) / (st.sum((1, 2, 3)) + eps)
    return (1 - 2 * tprec * tsens / (tprec + tsens)).mean()


def _voronoi_cells(fg: np.ndarray) -> tuple[np.ndarray, int]:
    """Cell id (1..K) of the nearest foreground component for every pixel; (zeros, 0) without foreground."""
    lab, n = ndi.label(fg, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(fg.shape, np.int64), 0
    _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    return lab[iy, ix].astype(np.int64), n


def bicc_loss(prob: torch.Tensor, tgt: torch.Tensor, alpha: float = 0.5, eps: float = 1.0) -> torch.Tensor:
    """Bidirectional connected-component loss: one equally weighted term per object, like PQ's RQ.

    Idea from BiCC (arXiv 2609.30223, as used in theainerd/filament_segmentation, MIT); our implementation
    for soft consensus targets. prob, tgt: (B, H, W).
      GT side: consensus components (tgt >= 0.5) split the tile into Voronoi cells; each cell adds
               1 - soft Dice(prob, tgt) inside the cell, so a small missed filament costs as much as a large one.
      FP side: each predicted blob (prob > 0.5) touching no consensus pixel adds its mean probability.
    loss = 2 * (alpha * sum(GT terms) + (1 - alpha) * sum(FP terms)) / (#GT + #FP)."""
    B = prob.shape[0]
    with torch.no_grad():  # component labelling on CPU, no gradient through it
        fg = (tgt >= 0.5).cpu().numpy()
        pb = (prob > 0.5).cpu().numpy()
        cells = np.zeros(fg.shape, np.int64)
        fps = np.zeros(fg.shape, np.int64)
        n_gt = n_fp = 0
        for b in range(B):
            c, k = _voronoi_cells(fg[b])
            cells[b] = np.where(c > 0, c + n_gt, 0)
            n_gt += k
            lab, nc = ndi.label(pb[b], structure=np.ones((3, 3)))
            if nc:
                keep = np.ones(nc + 1, bool)
                keep[0] = False
                keep[np.unique(lab[fg[b] & (lab > 0)])] = False
                remap = np.zeros(nc + 1, np.int64)
                remap[keep] = np.arange(1, keep.sum() + 1) + n_fp
                fps[b] = remap[lab]
                n_fp += int(keep.sum())
    if n_gt + n_fp == 0:
        return prob.sum() * 0
    p, t = prob.flatten(), tgt.flatten()
    total = prob.new_zeros(())
    if n_gt:
        ids = torch.from_numpy(cells).to(prob.device).flatten()
        z = prob.new_zeros(n_gt + 1)
        inter = z.index_add(0, ids, p * t)[1:]
        den = z.index_add(0, ids, p)[1:] + z.index_add(0, ids, t)[1:]
        total = total + alpha * (1 - (2 * inter + eps) / (den + eps)).sum()
    if n_fp:
        ids = torch.from_numpy(fps).to(prob.device).flatten()
        z = prob.new_zeros(n_fp + 1)
        s = z.index_add(0, ids, p)[1:]
        c = z.index_add(0, ids, torch.ones_like(p))[1:]
        total = total + (1 - alpha) * (s / c).sum()
    return 2 * total / (n_gt + n_fp)


class MultiHeadLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.heads = cfg["model"]["heads"]
        lc = cfg["loss"]["heads"]
        self.w = [lc.get(h, {}) for h in self.heads]
        self.cldice_iters = cfg["loss"].get("cldice_iters", 10)

    def forward(self, logits: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, dict]:
        logits = logits.float()
        total, parts = 0.0, {}
        for i, (h, w) in enumerate(zip(self.heads, self.w)):
            lg, t = logits[:, i], tgt[:, i]
            p = torch.sigmoid(lg)
            terms = {}
            if w.get("bce", 0):
                terms["bce"] = w["bce"] * F.binary_cross_entropy_with_logits(lg, t)
            if w.get("dice", 0):
                terms["dice"] = w["dice"] * soft_dice_loss(p, t)
            if w.get("cldice", 0):
                terms["cldice"] = w["cldice"] * soft_cldice_loss(p, t, self.cldice_iters)
            if w.get("bicc", 0):
                terms["bicc"] = w["bicc"] * bicc_loss(p, t)
            for k, v in terms.items():
                parts[f"{h}_{k}"] = v.detach()
                total = total + v
        return total, parts
