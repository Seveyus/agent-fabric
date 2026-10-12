"""Leakage and correctness tests for core/dataset/temporal_graph.py."""

from __future__ import annotations

import numpy as np
import pytest

pl = pytest.importorskip("polars")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from core.dataset.synthetic_events import SyntheticConfig, generate_events  # noqa: E402
from core.dataset.temporal_graph import (  # noqa: E402
    EDGE_TYPES,
    PR_FEATURES,
    PRMergeTaskConfig,
    TemporalTask,
    build_pr_merge_task,
)

CFG = PRMergeTaskConfig(
    train_start="2025-09-05", train_end="2025-09-09", val_start="2025-09-11", val_end="2025-09-12",
    test_start="2025-09-14", test_end="2025-09-15", horizon_hours=48,
)


@pytest.fixture(scope="module")
def events():
    return generate_events(SyntheticConfig(n_users=300, n_repos=30, days=16, seed=1))


@pytest.fixture(scope="module")
def task(events):
    return build_pr_merge_task(events, CFG)


def test_schema(task):
    assert set(task.data.edge_types) == set(EDGE_TYPES)
    assert task.data["pr"].x.shape[1] == len(PR_FEATURES)
    for et in task.data.edge_types:
        store = task.data[et]
        assert store.time.shape[0] == store.edge_index.shape[1]
        if store.edge_index.numel():
            s, _, d = et
            assert int(store.edge_index[0].max()) < task.data[s].num_nodes
            assert int(store.edge_index[1].max()) < task.data[d].num_nodes


def test_embargo_validation():
    with pytest.raises(ValueError, match="embargo"):
        PRMergeTaskConfig(train_start="2025-09-01", train_end="2025-09-05", val_start="2025-09-06",
                          val_end="2025-09-07", test_start="2025-09-10", test_end="2025-09-11",
                          horizon_hours=48).validate()


def test_splits_respect_time_bounds(task):
    for name, (start, end) in CFG.bounds().items():
        t = task.splits[name].time
        assert len(t) > 0
        assert int(t.min()) >= start and int(t.max()) < end


def _pr_key_map(events):
    """(repo, number) -> pr_id, using the same deterministic ordering as the builder."""
    keys = (events.filter(pl.col("number").is_not_null() & pl.col("etype").str.starts_with("pr_"))
            .select("repo", "number").unique().sort("repo", "number"))
    return {(r, n): i for i, (r, n) in enumerate(keys.iter_rows())}


def test_labels_match_brute_force(events, task):
    key_of = {v: k for k, v in _pr_key_map(events).items()}
    closes = events.filter((pl.col("etype") == "pr_close") & pl.col("merged"))
    merge_ts = {}
    for r, n, ts in closes.select("repo", "number", "ts").iter_rows():
        merge_ts[(r, n)] = min(ts, merge_ts.get((r, n), ts))
    s = task.splits["train"]
    for node, t_open, y in zip(s.node.tolist(), s.time.tolist(), s.y.tolist()):
        m = merge_ts.get(key_of[node])
        expected = float(m is not None and t_open < m <= t_open + CFG.horizon_s)
        assert y == expected


def test_history_features_are_point_in_time(events, task):
    """Brute-force check that history aggregates only count events strictly before the open time."""
    key_of = {v: k for k, v in _pr_key_map(events).items()}
    opens = events.filter(pl.col("etype") == "pr_open")
    closes = events.filter(pl.col("etype") == "pr_close")
    idx = {f: PR_FEATURES.index(f) for f in PR_FEATURES}
    x = task.data["pr"].x
    rng = np.random.default_rng(0)
    nodes = task.splits["test"].node.numpy()
    for node in rng.choice(nodes, size=min(40, len(nodes)), replace=False):
        repo, number = key_of[int(node)]
        row = opens.filter((pl.col("repo") == repo) & (pl.col("number") == number)).row(0, named=True)
        t, author = row["ts"], row["pr_author"]
        before = closes.filter(pl.col("ts") < t)
        n_merged = before.filter(pl.col("merged") & (pl.col("pr_author") == author)).height
        n_closed = before.filter(~pl.col("merged") & (pl.col("pr_author") == author)).height
        n_opened = opens.filter((pl.col("ts") < t) & (pl.col("pr_author") == author)).height
        n_repo_merged = before.filter(pl.col("merged") & (pl.col("repo") == repo)).height
        assert x[node, idx["log_author_prior_merged"]].item() == pytest.approx(np.log1p(n_merged), abs=1e-5)
        assert x[node, idx["log_author_prior_closed"]].item() == pytest.approx(np.log1p(n_closed), abs=1e-5)
        assert x[node, idx["log_author_prior_opened"]].item() == pytest.approx(np.log1p(n_opened), abs=1e-5)
        assert x[node, idx["log_repo_prior_merged"]].item() == pytest.approx(np.log1p(n_repo_merged), abs=1e-5)
        assert x[node, idx["author_merge_rate"]].item() == pytest.approx((n_merged + 1) / (n_merged + n_closed + 2), abs=1e-5)


def test_no_bot_seeds(task):
    bot_col = PR_FEATURES.index("author_is_bot")
    for s in task.splits.values():
        assert float(task.data["pr"].x[s.node, bot_col].sum()) == 0.0


def test_seed_subsampling_is_deterministic(events):
    import dataclasses
    cfg = dataclasses.replace(CFG, max_train_seeds=50, sample_seed=3)
    a = build_pr_merge_task(events, cfg).splits["train"].node
    b = build_pr_merge_task(events, cfg).splits["train"].node
    assert len(a) == 50 and torch.equal(a, b)


def test_save_load_roundtrip(task, tmp_path):
    path = tmp_path / "task.pt"
    task.save(path)
    loaded = TemporalTask.load(path)
    assert torch.equal(loaded.data["pr"].x, task.data["pr"].x)
    assert torch.equal(loaded.splits["val"].y, task.splits["val"].y)
    et = ("user", "opens", "pr")
    assert torch.equal(loaded.data[et].time, task.data[et].time)


def test_build_is_deterministic(events):
    """Edge order (hence the CSC neighbour order the sampler sees) must not depend on
    polars tie-breaking; otherwise identical seeds sample different neighbours."""
    a = build_pr_merge_task(events, CFG).data
    b = build_pr_merge_task(events.sample(fraction=1.0, shuffle=True, seed=7), CFG).data
    for et in a.edge_types:
        assert torch.equal(a[et].edge_index, b[et].edge_index), et
        assert torch.equal(a[et].time, b[et].time), et
    assert torch.equal(a["pr"].x, b["pr"].x)
