"""Alternative candidate-scorer formulation: P(match) x E[IoU | match].

The default scorer regresses the target y = mean over annotators of IoU*[IoU>0.5]
directly. This model factorises it:
  * a classifier on one row per (candidate, annotator) with binary label
    "IoU > 0.5 with that annotator's best GT" (log-loss, calibrated)
  * a regressor of that IoU on the matched rows only
predict(X) = P(match) * E[IoU | match], the same quantity, so selection
(tree DP with tau) is unchanged.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


class HitIoUModel:
    def __init__(self, gbm: dict):
        self.gbm = gbm

    def fit(self, X: np.ndarray, hit: np.ndarray, iou: np.ndarray, row_cand: np.ndarray):
        """X: (n_cand, F). hit/iou/row_cand: one entry per (candidate, annotator) row."""
        Xr = X[row_cand]
        self.clf = HistGradientBoostingClassifier(**self.gbm).fit(Xr, hit.astype(int))
        pos = hit.astype(bool)
        self.reg = HistGradientBoostingRegressor(**self.gbm).fit(Xr[pos], iou[pos])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.clf.predict_proba(X)[:, 1]
        q = np.clip(self.reg.predict(X), 0.5, 1.0)
        return p * q


def annotator_rows(frames: list[dict], offs: np.ndarray):
    """Per-(candidate, annotator) labels from the cached per-annotator IoU matrices."""
    hit, iou, row = [], [], []
    for i, f in enumerate(frames):
        n = offs[i + 1] - offs[i]
        for m in f["iou"]:
            best = m.max(0) if m.shape[0] else np.zeros(n)
            hit.append(best > 0.5)
            iou.append(best)
            row.append(np.arange(offs[i], offs[i + 1]))
    return np.concatenate(hit), np.concatenate(iou), np.concatenate(row)
