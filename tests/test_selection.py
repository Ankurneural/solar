"""Candidate tree, tree-DP selection and post-processing invariants.

  python -m pytest tests -q
"""
from __future__ import annotations

import itertools

import cv2
import numpy as np

from src.candidates import generate, select
from src.postprocess import instances_from_probs, to_working_units

CAND = {"levels": [0.2, 0.4, 0.6, 0.8], "close_px": 0, "merge_px": 0, "min_area": 0}


def _prob_map(size=256, seed=0):
    """Blobs with internal structure, so higher thresholds split them into children."""
    rng = np.random.default_rng(seed)
    p = np.zeros((size, size), np.float32)
    for _ in range(5):
        cx, cy = rng.integers(30, size - 30, 2)
        for _ in range(3):
            dx, dy = rng.integers(-12, 12, 2)
            cv2.circle(p, (int(cx + dx), int(cy + dy)), int(rng.integers(4, 10)),
                       float(rng.uniform(0.5, 1.0)), -1)
    return cv2.GaussianBlur(p, (0, 0), 3) * 1.6


def _cands(seed=0):
    probs = {"mask": _prob_map(seed=seed)}
    return generate(probs, None, None, None, CAND)


def test_tree_is_nested():
    cands = _cands()
    S = 256
    for i, c in enumerate(cands):
        if c.parent >= 0:
            par = cands[c.parent]
            assert par.level == c.level - 1
            assert i in par.children
            assert not (c.full(S) & ~par.full(S)).any(), "child must lie inside its parent"


def _valid(cands, chosen):
    """At most one node per root-to-leaf path."""
    def ancestors(i):
        while cands[i].parent >= 0:
            i = cands[i].parent
            yield i
    s = set(chosen)
    return all(not (set(ancestors(i)) & s) for i in chosen)


def _subtree(cands, root):
    out, stack = [], [root]
    while stack:
        i = stack.pop()
        out.append(i)
        stack += cands[i].children
    return out


def test_select_matches_brute_force():
    """Tree DP == exhaustive search. Different root trees are independent, so each
    root's subtree is brute-forced on its own (all subsets of its nodes)."""
    n_checked = 0
    for seed in range(6):
        cands = _cands(seed)
        rng = np.random.default_rng(seed)
        scores = rng.uniform(0, 0.6, len(cands))
        tau = 0.25
        got = select(cands, scores, tau)
        assert _valid(cands, got)
        for root in (i for i, c in enumerate(cands) if c.parent < 0):
            nodes = _subtree(cands, root)
            assert len(nodes) <= 16, "keep the brute force tractable"
            best = 0.0
            for r in range(1, len(nodes) + 1):
                for comb in itertools.combinations(nodes, r):
                    if _valid(cands, comb):
                        best = max(best, sum(scores[i] - tau for i in comb))
            mine = sum(scores[i] - tau for i in got if i in set(nodes))
            assert abs(mine - best) < 1e-9
            n_checked += 1
    assert n_checked >= 10


def test_selected_candidates_are_disjoint():
    cands = _cands(3)
    sel = select(cands, np.full(len(cands), 0.9), tau=0.1)
    acc = np.zeros((256, 256), np.int32)
    for i in sel:
        acc += cands[i].full(256)
    assert acc.max() <= 1


def test_pixel_units_are_resolution_independent():
    p = {"close_px": 8, "merge_px": 20, "grow_px": 4, "min_area": 600}
    q = to_working_units(p, 1024)
    assert (q["close_px"], q["merge_px"], q["grow_px"]) == (4, 10, 2)
    assert abs(q["min_area"] - 150) < 1e-9
    assert to_working_units(p, 2048)["min_area"] == 600


def test_instances_disjoint_and_filtered():
    prob = _prob_map(512, seed=1)
    pp = {"threshold": 0.5, "close_px": 0, "merge_px": 0, "min_area": 200, "min_score": 0.0,
          "grow_px": 12}
    insts = instances_from_probs({"mask": prob}, None, pp)
    acc = np.zeros(prob.shape, np.int32)
    for t in insts:
        acc += t.mask
        assert t.area >= pp["min_area"] * (512 / 2048) ** 2
    assert acc.max() <= 1, "grown instances must still be disjoint"
