"""
Paired comparison of two models evaluated on the same test seeds.

    python -m training.compare runs/predictions/gh_pr_merge__main "HGT (+history)" "LightGBM (+history)"

For B bootstrap resamples of the test seeds (the same indices for both models),
compute mean over model seeds of AUC(A) - AUC(B). Reports the mean difference
and a 95% percentile interval. Resampling seeds jointly keeps the pairing, so
the interval is much tighter than comparing two unpaired CIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def _slug(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label).strip("_").lower()


def load_predictions(pred_dir: str | Path, label: str) -> tuple[np.ndarray, list[np.ndarray]]:
    files = sorted(Path(pred_dir).glob(f"{_slug(label)}__seed*.npz"))
    if not files:
        raise FileNotFoundError(f"no predictions for {label!r} in {pred_dir}")
    ys, ps = [], []
    for f in files:
        z = np.load(f)
        ys.append(z["y"])
        ps.append(z["p"])
    y = ys[0]
    if any(not np.array_equal(y, other) for other in ys[1:]):
        raise ValueError(f"{label}: test labels differ across seeds (seeds not in the same order?)")
    return y, ps


def paired_auc_difference(y: np.ndarray, pa: list[np.ndarray], pb: list[np.ndarray],
                          n_boot: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)

    def diff(idx: np.ndarray) -> float:
        yy = y[idx]
        a = np.mean([roc_auc_score(yy, p[idx]) for p in pa])
        b = np.mean([roc_auc_score(yy, p[idx]) for p in pb])
        return a - b

    full = diff(np.arange(n))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if 0 < y[idx].sum() < n:
            boots.append(diff(idx))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"auc_diff": float(full), "ci95": [float(lo), float(hi)],
            "p_le_0": float(np.mean(np.array(boots) <= 0)), "n_test": int(n),
            "n_seeds": [len(pa), len(pb)]}


def compare(pred_dir: str | Path, a: str, b: str, n_boot: int = 1000) -> dict:
    ya, pa = load_predictions(pred_dir, a)
    yb, pb = load_predictions(pred_dir, b)
    if not np.array_equal(ya, yb):
        raise ValueError("the two models were not evaluated on the same test seeds in the same order")
    return {"a": a, "b": b, **paired_auc_difference(ya, pa, pb, n_boot=n_boot)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pred_dir")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args(argv)
    print(json.dumps(compare(args.pred_dir, args.a, args.b, args.n_boot), indent=2))


if __name__ == "__main__":
    main()
