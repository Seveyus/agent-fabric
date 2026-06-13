"""
Outcome labeler for JEPA training pairs.

Given two consecutive snapshots (T and T+k), produces:
  - regression_label   : 1 if the T+k window contains a revert/regression
  - delay_label        : 1 if release cadence dropped meaningfully
  - velocity_delta     : continuous — change in merge velocity (normalized)

These labels are used both for supervised downstream heads and to
validate that the world model learns meaningful representations.
"""

from __future__ import annotations

import logging

import numpy as np

from core.dataset.graph_builder import Snapshot

log = logging.getLogger(__name__)

REVERT_KEYWORDS = {"revert", "rollback", "undo", "hotfix", "fix regression", "fix: revert"}


def label_pair(snap_t: Snapshot, snap_t_k: Snapshot) -> dict:
    """
    Compute supervision labels for a (T, T+k) snapshot pair.

    Returns a dict with:
        regression      int   1/0 — revert or regression in T+k window
        velocity_drop   int   1/0 — merge velocity dropped >40% from T to T+k
        release_in_t_k  int   1/0 — a release was published in T+k
        velocity_delta  float normalized delta in PR merge rate
    """
    # Regression: revert PRs or hotfixes in T+k
    regression = snap_t_k.label_is_regression

    # Velocity: PRs merged per day
    def merge_rate(snap: Snapshot) -> float:
        days = max(1.0, (snap.t_end - snap.t_start).days)
        merged = sum(1 for ev in snap.events if ev["type"] == "pr_merged")
        return merged / days

    rate_t = merge_rate(snap_t)
    rate_t_k = merge_rate(snap_t_k)
    delta = rate_t_k - rate_t
    normalized_delta = np.tanh(delta)  # squash to [-1, 1]
    velocity_drop = int(rate_t > 0 and rate_t_k < rate_t * 0.6)

    # Release
    release_in_t_k = int(any(ev["type"] == "release_published" for ev in snap_t_k.events))

    return {
        "regression": regression,
        "velocity_drop": velocity_drop,
        "release_in_t_k": release_in_t_k,
        "velocity_delta": float(normalized_delta),
    }


def compute_dataset_stats(labels: list[dict]) -> dict:
    """Compute class balance and sanity stats over the full dataset."""
    n = len(labels)
    if n == 0:
        return {}
    return {
        "n_samples": n,
        "regression_rate": sum(lb["regression"] for lb in labels) / n,
        "velocity_drop_rate": sum(lb["velocity_drop"] for lb in labels) / n,
        "release_rate": sum(lb["release_in_t_k"] for lb in labels) / n,
        "mean_velocity_delta": float(np.mean([lb["velocity_delta"] for lb in labels])),
    }
