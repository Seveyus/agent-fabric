"""
Non-graph baselines on exactly the same seeds, splits and metrics as the GNNs.

    logreg   standardised logistic regression
    lgbm     LightGBM, early-stopped on validation ROC-AUC (optional dependency)

Inputs are the seed PR's own feature vector — intrinsic open-time features,
optionally plus the point-in-time history aggregates. With history, these
baselines already use relational information (author / repo track record)
through feature engineering; the question for the GNN is whether learning
from the raw temporal graph adds anything on top.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from core.dataset.temporal_graph import TemporalTask
from training.metrics import best_f1_threshold, binary_metrics, bootstrap_auc_ci


def run_tabular(task: TemporalTask, model: str, use_history: bool, seed: int = 0,
                label_fraction: float = 1.0) -> dict:
    Xtr, ytr = task.tabular("train", with_history=use_history)
    Xva, yva = task.tabular("val", with_history=use_history)
    Xte, yte = task.tabular("test", with_history=use_history)
    if label_fraction < 1.0:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.permutation(len(ytr))[: max(1, int(round(len(ytr) * label_fraction)))])
        Xtr, ytr = Xtr[keep], ytr[keep]

    t0 = time.perf_counter()
    if model == "logreg":
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(scaler.transform(Xtr), ytr)

        def predict(X):
            return clf.predict_proba(scaler.transform(X))[:, 1]
        n_params = int(clf.coef_.size + clf.intercept_.size)
    elif model == "lgbm":
        import lightgbm as lgb

        params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 63,
                  "min_data_in_leaf": 50, "feature_fraction": 0.9, "bagging_fraction": 0.9,
                  "bagging_freq": 1, "seed": seed, "deterministic": True, "num_threads": 4, "verbose": -1}
        booster = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=2000,
                            valid_sets=[lgb.Dataset(Xva, yva)],
                            callbacks=[lgb.early_stopping(50, verbose=False)])

        def predict(X):
            return booster.predict(X, num_iteration=booster.best_iteration)
        n_params = int(booster.best_iteration)  # number of trees
    else:
        raise ValueError(f"unknown tabular model {model!r}")
    train_seconds = time.perf_counter() - t0

    pva = predict(Xva)
    t1 = time.perf_counter()
    pte = predict(Xte)
    infer = time.perf_counter() - t1
    thr = best_f1_threshold(yva, pva)
    test = binary_metrics(yte, pte, thr)
    test["roc_auc_ci95"] = bootstrap_auc_ci(yte, pte, seed=seed)
    return {
        "model": model,
        "use_history": use_history,
        "label_fraction": label_fraction,
        "trainable_parameters": n_params,
        "val": binary_metrics(yva, pva, thr),
        "test": test,
        "timing": {"train_seconds_total": train_seconds, "test_inference_seconds": infer,
                   "test_seeds_per_second": len(yte) / max(infer, 1e-9)},
        "_test_pred": (yte, pte),  # in-memory only (used by training/suite.py)
    }
