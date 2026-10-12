import numpy as np
import pytest

from training.compare import compare, paired_auc_difference
from training.metrics import best_f1_threshold, binary_metrics


def test_paired_difference_detects_better_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    good = [y + rng.normal(0, 0.8, len(y)) for _ in range(2)]
    bad = [y + rng.normal(0, 3.0, len(y)) for _ in range(2)]
    r = paired_auc_difference(y, good, bad, n_boot=200)
    assert r["auc_diff"] > 0 and r["ci95"][0] > 0 and r["p_le_0"] == 0.0
    same = paired_auc_difference(y, good, good, n_boot=50)
    assert same["auc_diff"] == 0.0


def test_compare_requires_same_test_set(tmp_path):
    y = np.array([0, 1, 0, 1])
    np.savez(tmp_path / "a__seed0.npz", y=y, p=np.array([0.1, 0.9, 0.2, 0.8]))
    np.savez(tmp_path / "b__seed0.npz", y=y[::-1], p=np.array([0.1, 0.9, 0.2, 0.8]))
    with pytest.raises(ValueError):
        compare(tmp_path, "a", "b")


def test_threshold_is_chosen_on_given_split():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    thr = best_f1_threshold(y, p)
    m = binary_metrics(y, p, thr)
    assert m["f1"] == pytest.approx(0.8) and m["roc_auc"] == pytest.approx(0.75)
