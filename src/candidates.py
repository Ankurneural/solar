"""Candidate tree + learned selection: a metric-aware replacement for a
single threshold.

Why: PQ rewards a candidate only if it is worth more than it costs. Keeping a
candidate adds (expected) p*q to the numerator and always 1/2 to the
denominator, so it should be kept only if p*q > PQ/2 (docs/COMPETITION_NOTES.md).
A single threshold + min_area can't express that: faint filaments need a low
threshold, while high thresholds separate neighbours that touch at low ones.

1. **Candidates.** Threshold the mask probability at several levels. Each
   level's connected components (grouped as in postprocess `cc`) are candidates.
   Because P > t_high implies P > t_low, components nest into a tree: each
   candidate's parent is the component one level lower that contains it.
2. **Features** per candidate: probability statistics, spine/boundary head
   support, shape (area, elongation, solidity, compactness), position on the
   disk, image contrast with its surroundings, and tree context (#children,
   parent area ratio).
3. **Target**, averaged over the frame's annotators, each scored separately as
   in the metric: IoU with that annotator's best-matching GT if IoU > 0.5, else 0.
   This is exactly the candidate's expected contribution to the PQ numerator.
4. **Selection.** An exact dynamic program on the tree keeps at most one
   candidate per root-to-leaf path and maximises sum(score - tau). Any two
   kept candidates are on different branches, so they are disjoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi

from .postprocess import NATIVE, _disk, _group_cc, to_working_units

FEATURES = [
    "level", "area", "log_area", "mean_prob", "max_prob", "p25_prob", "p75_prob", "std_prob",
    "spine_mean", "spine_max", "spine_cover", "boundary_mean",
    "perimeter", "compactness", "elongation", "long_side", "short_side", "solidity",
    "r_disk", "inside_int", "ring_int", "contrast",
    "n_children", "parent_area_ratio", "is_root", "n_at_level", "prob_rank", "site",
    # boundary sharpness: probability just outside the candidate, near and far
    "edge_prob_out", "far_prob_out", "edge_drop",
    # image texture / contrast at two radii
    "inside_p10", "inside_std", "far_ring_int", "far_contrast",
    # tree context
    "children_area_frac", "max_child_frac", "parent_mean_prob", "n_siblings", "depth_below",
    # heliographic position (approx., no B0/P tilt): annotators skipped filaments beyond ~±70° from the
    # central meridian (MAGFiLO protocol), which the radial r_disk cannot tell apart from the poles
    "abs_lon", "abs_lat",
]
SITES = {"Bh": 0, "Ch": 1, "Lh": 2, "Mh": 3, "Th": 4, "Uh": 5}
# Ensemble-member mask maps can be passed in probs as "member0", "member1", ...
# (in ensemble.yaml order). They add per-member mean/max probability inside the
# candidate plus the spread across members: disagreement between a native-res
# crop model and the full-disk model is informative about false positives.
MEMBER_PREFIX = "member"


def member_keys(probs: dict) -> list[str]:
    return sorted((k for k in probs if k.startswith(MEMBER_PREFIX)), key=lambda k: int(k[len(MEMBER_PREFIX):]))


def feature_names(n_members: int) -> list[str]:
    extra = [f"m{i}_{s}" for i in range(n_members) for s in ("mean", "max")]
    return FEATURES + (extra + ["member_std", "member_min"] if n_members else [])


@dataclass
class Candidate:
    y0: int
    x0: int
    mask: np.ndarray             # bool crop
    level: int
    parent: int = -1
    children: list = field(default_factory=list)
    feats: dict = field(default_factory=dict)

    def full(self, size: int) -> np.ndarray:
        out = np.zeros((size, size), bool)
        h, w = self.mask.shape
        out[self.y0:self.y0 + h, self.x0:self.x0 + w] = self.mask
        return out


def generate(probs: dict[str, np.ndarray], disk_mask: np.ndarray | None, img: np.ndarray | None,
             disk_rc: tuple[float, float, float] | None, p: dict, site: str = "") -> list[Candidate]:
    """probs at working size S (float); img = preprocessed uint8 frame at S or None;
    disk_rc = (cx, cy, r) at S. p: candidate params (native units)."""
    prob = probs["mask"]
    S = prob.shape[0]
    q = to_working_units(p, S)
    s = S / NATIVE
    spine, bnd = probs.get("spine"), probs.get("boundary")
    members = [probs[k] for k in member_keys(probs)]
    ring_k = (_disk(max(1, int(round(4 * s)))), _disk(max(2, int(round(24 * s)))))
    cands: list[Candidate] = []
    prev_lab, prev_map = None, {}
    for lev, t in enumerate(p["levels"]):
        b = prob > t
        if disk_mask is not None:
            b &= disk_mask
        lab = _group_cc(b, q.get("close_px", 0), q.get("merge_px", 0))
        this_map = {}
        level_ids = []
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            if sl is None:
                continue
            m = lab[sl] == i
            area = int(m.sum())
            if area < q.get("min_area", 0):
                continue
            parent = -1
            if prev_lab is not None:
                # Any pixel of this component lies inside its parent (P > t_high => P > t_low).
                yy, xx = np.nonzero(m)
                pl = prev_lab[sl[0].start + yy[0], sl[1].start + xx[0]]
                parent = prev_map.get(int(pl), -1)
            c = Candidate(sl[0].start, sl[1].start, ndi.binary_fill_holes(m), lev, parent)
            idx = len(cands)
            cands.append(c)
            if parent >= 0:
                cands[parent].children.append(idx)
            this_map[i] = idx
            level_ids.append(idx)
            c.feats = _features(c, prob, spine, bnd, img, disk_rc, ring_k, s, t)
            c.feats["site"] = SITES.get(site, -1)
            if members:
                h, w = c.mask.shape
                means = []
                for mi, mp in enumerate(members):
                    v = mp[c.y0:c.y0 + h, c.x0:c.x0 + w][c.mask]
                    c.feats[f"m{mi}_mean"], c.feats[f"m{mi}_max"] = float(v.mean()), float(v.max())
                    means.append(c.feats[f"m{mi}_mean"])
                c.feats["member_std"], c.feats["member_min"] = float(np.std(means)), float(min(means))
        for rank, idx in enumerate(sorted(level_ids, key=lambda k: -cands[k].feats["mean_prob"])):
            cands[idx].feats["n_at_level"] = len(level_ids)
            cands[idx].feats["prob_rank"] = rank
        prev_lab, prev_map = lab, this_map
    depth = [0] * len(cands)
    for i in sorted(range(len(cands)), key=lambda k: -cands[k].level):  # leaves first
        depth[i] = 1 + max((depth[j] for j in cands[i].children), default=-1)
    for i, c in enumerate(cands):
        f = c.feats
        kids = [cands[j].feats["area"] for j in c.children]
        par = cands[c.parent] if c.parent >= 0 else None
        f["n_children"] = len(c.children)
        f["is_root"] = int(par is None)
        f["parent_area_ratio"] = f["area"] / par.feats["area"] if par else 1.0
        f["parent_mean_prob"] = par.feats["mean_prob"] if par else 0.0
        f["n_siblings"] = len(par.children) - 1 if par else 0
        f["children_area_frac"] = sum(kids) / f["area"]
        f["max_child_frac"] = max(kids, default=0.0) / f["area"]
        f["depth_below"] = depth[i]
    return cands


def _features(c: Candidate, prob, spine, bnd, img, disk_rc, ring_k, s: float, t: float) -> dict:
    h, w = c.mask.shape
    ys, xs = slice(c.y0, c.y0 + h), slice(c.x0, c.x0 + w)
    m = c.mask
    pv = prob[ys, xs][m]
    area = float(m.sum())
    f = {"level": t, "area": area / s / s, "log_area": np.log1p(area / s / s),
         "mean_prob": float(pv.mean()), "max_prob": float(pv.max()),
         "p25_prob": float(np.percentile(pv, 25)), "p75_prob": float(np.percentile(pv, 75)),
         "std_prob": float(pv.std())}
    if spine is not None:
        sv = spine[ys, xs][m]
        f.update(spine_mean=float(sv.mean()), spine_max=float(sv.max()),
                 spine_cover=float((sv > 0.5).mean()))
    else:
        f.update(spine_mean=0.0, spine_max=0.0, spine_cover=0.0)
    f["boundary_mean"] = float(bnd[ys, xs][m].mean()) if bnd is not None else 0.0
    m8 = m.astype(np.uint8)
    cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cs, key=cv2.contourArea)
    per = cv2.arcLength(cnt, True)
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    hull = cv2.contourArea(cv2.convexHull(cnt))
    f.update(perimeter=per / s, compactness=4 * np.pi * area / max(per * per, 1),
             elongation=max(rw, rh) / max(min(rw, rh), 1), long_side=max(rw, rh) / s,
             short_side=min(rw, rh) / s, solidity=area / max(hull, 1))
    cy, cx = c.y0 + h / 2, c.x0 + w / 2
    f["r_disk"] = (float(np.hypot(cx - disk_rc[0], cy - disk_rc[1]) / disk_rc[2])
                   if disk_rc else 0.0)
    if disk_rc:
        u, v = (cx - disk_rc[0]) / disk_rc[2], (cy - disk_rc[1]) / disk_rc[2]
        f["abs_lat"] = float(np.degrees(np.arcsin(min(1.0, abs(v)))))
        f["abs_lon"] = float(np.degrees(np.arcsin(min(1.0, abs(u) / max(1e-6, np.sqrt(max(0.0, 1 - v * v)))))))
    else:
        f["abs_lat"] = f["abs_lon"] = 0.0
    # Rings around the candidate (native 4 px and 24 px), on the probability and the image.
    near_k, far_k = ring_k
    pad = far_k.shape[0]
    H, W = prob.shape
    y0, x0 = max(0, c.y0 - pad), max(0, c.x0 - pad)
    y1, x1 = min(H, c.y0 + h + pad), min(W, c.x0 + w + pad)
    big = np.zeros((y1 - y0, x1 - x0), np.uint8)
    big[c.y0 - y0:c.y0 - y0 + h, c.x0 - x0:c.x0 - x0 + w] = m8
    near = (cv2.dilate(big, near_k) > 0) & (big == 0)
    far = (cv2.dilate(big, far_k) > 0) & (big == 0) & ~near
    pc = prob[y0:y1, x0:x1]
    e_out = float(pc[near].mean()) if near.any() else 0.0
    f.update(edge_prob_out=e_out, far_prob_out=float(pc[far].mean()) if far.any() else 0.0,
             edge_drop=f["mean_prob"] - e_out)
    if img is not None:
        crop = img[y0:y1, x0:x1].astype(np.float32)
        iv = crop[big > 0]
        inside = float(iv.mean())
        on = crop > 0
        nv, fv = crop[near & on], crop[far & on]
        ring_i = float(nv.mean()) if nv.size else inside
        far_i = float(fv.mean()) if fv.size else ring_i
        f.update(inside_int=inside, ring_int=ring_i, contrast=ring_i - inside,
                 inside_p10=float(np.percentile(iv, 10)), inside_std=float(iv.std()),
                 far_ring_int=far_i, far_contrast=far_i - inside)
    else:
        f.update(inside_int=0.0, ring_int=0.0, contrast=0.0, inside_p10=0.0, inside_std=0.0,
                 far_ring_int=0.0, far_contrast=0.0)
    return f


def select(cands: list[Candidate], scores: np.ndarray, tau: float) -> list[int]:
    """Exact tree DP: max sum(score - tau), at most one node per root-leaf path."""
    best = np.zeros(len(cands))
    take = np.zeros(len(cands), bool)
    for i in sorted(range(len(cands)), key=lambda k: -cands[k].level):  # leaves first
        own = scores[i] - tau
        kids = sum(best[j] for j in cands[i].children)
        take[i] = own > kids and own > 0
        best[i] = max(own, kids, 0.0)
    out = []

    def collect(i):
        if take[i]:
            out.append(i)
        else:
            for j in cands[i].children:
                collect(j)

    for i, c in enumerate(cands):
        if c.parent < 0:
            collect(i)
    return out


def feature_matrix(cands: list[Candidate], names: list[str] | None = None) -> np.ndarray:
    names = names or FEATURES
    return np.array([[c.feats[k] for k in names] for c in cands], np.float32).reshape(-1, len(names))


def predict_with_scorer(probs: dict[str, np.ndarray], img_u8: np.ndarray, disk, scorer: dict,
                        site: str, out_size: int = NATIVE, disk_margin: float = 0.01,
                        refiners=None, device: str = "cpu", blend: float = 1.0,
                        tau: float | None = None) -> list[np.ndarray]:
    """Full learned-selection inference for one frame -> disjoint bool masks at out_size.

    Candidates are built at the scorer's working size (as in training). Kept ones
    are either redrawn by the crop refiners (if given) or upsampled smoothly, then
    painted in score order. With refiners and blend < 1, every candidate is also
    scored by the refiners and the selection uses blend*GBM + (1-blend)*refiner
    (tuned by tune_stack.py)."""
    from .evaluation import disk_mask_at, resize_probs

    S = scorer["eval_size"]
    probs = resize_probs(probs, S)
    img = cv2.resize(img_u8, (S, S), interpolation=cv2.INTER_AREA)
    s = S / NATIVE
    disk_rc = (disk.cx * s, disk.cy * s, disk.r * s) if disk is not None else None
    cands = generate(probs, disk_mask_at(disk, S, disk_margin), img, disk_rc, scorer["cand"], site)
    if not cands:
        return []
    scores = scorer["model"].predict(feature_matrix(cands, scorer.get("features")))
    tau = scorer["tau"] if tau is None else tau
    if refiners:
        from .refiner import refined_native, run_refiners

        k = NATIVE / S

        def bbox(c):
            return (c.x0 * k, c.y0 * k, (c.x0 + c.mask.shape[1]) * k, (c.y0 + c.mask.shape[0]) * k)

        # Refine every candidate when its refiner score is needed, else only the kept ones.
        todo = (list(range(len(cands))) if blend < 1 else
                select(cands, scores, tau))
        crops, rscores, boxes = run_refiners(refiners, img_u8, probs["mask"],
                                             [cands[i].full(S) for i in todo],
                                             [bbox(cands[i]) for i in todo], device)
        at = {i: n for n, i in enumerate(todo)}
        if blend < 1:
            scores = blend * scores + (1 - blend) * rscores
    keep = sorted(select(cands, scores, tau), key=lambda i: -scores[i])
    if refiners:
        refined = [refined_native(crops[at[i]], boxes[at[i]]) for i in keep]
    taken = np.zeros((out_size, out_size), bool)
    out = []
    for n, i in enumerate(keep):
        if refiners:
            m = refined[n]
        else:
            m = cands[i].full(S).astype(np.float32)
            if S != out_size:
                m = cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            m = m > 0.5
        m = m & ~taken
        if m.sum() >= 50:  # pixels lost to a higher-scoring neighbour can leave slivers
            taken |= m
            out.append(m)
    return out
