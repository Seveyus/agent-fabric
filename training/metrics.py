"""Binary classification metrics with a threshold chosen on validation data only."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising F1 on (y, p). Call on the *validation* split only."""
    if len(np.unique(y)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= threshold).astype(int)
    two_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "pos_rate": float(y.mean()) if len(y) else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "avg_precision": float(average_precision_score(y, p)) if two_classes else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(threshold),
    }


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 500, seed: int = 0,
                     alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for ROC-AUC (resampling seeds with replacement)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    aucs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    if not aucs:
        return float("nan"), float("nan")
    lo, hi = np.quantile(aucs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
