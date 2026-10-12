"""Synthetic EV fleet example: graph schema, splits and point-in-time features."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from core.dataset.ev_fleet import (  # noqa: E402
    AGGREGATE_FEATURES,
    EDGE_TYPES,
    SESSION_FEATURES,
    EVFleetConfig,
    build_ev_task,
    simulate,
)

CFG = EVFleetConfig(n_feeders=2, nights=12, train_nights=6, val_nights=3, history_nights=5, seed=1)


@pytest.fixture(scope="module")
def task():
    return build_ev_task(CFG)


def test_schema_and_metadata(task):
    assert task.seed_type == "session"
    node_types, edge_types = task.metadata
    assert set(edge_types) == set(EDGE_TYPES) == set(task.data.edge_types)
    assert task.data["session"].x.shape[1] == len(SESSION_FEATURES)
    assert task.history_features == AGGREGATE_FEATURES


def test_splits_are_by_night(task):
    day = 86400
    nights = {k: set(((v.time - v.time.min()) // day).tolist()) for k, v in task.splits.items()}
    t_max_train = int(task.splits["train"].time.max())
    assert t_max_train < int(task.splits["val"].time.min())
    assert int(task.splits["val"].time.max()) < int(task.splits["test"].time.min())
    assert all(len(n) >= 1 for n in nights.values())


def test_night_context_nodes_only_link_same_night(task):
    ei = task.data["session", "in", "depot_night"]
    t = ei.time
    # every session -> depot_night edge carries the session's own planning time
    seed_t = torch.empty(task.data["session"].num_nodes, dtype=torch.long)
    for s in task.splits.values():
        seed_t[s.node] = s.time
    assert torch.equal(t, seed_t[ei.edge_index[0]])


def test_charger_fault_history_is_point_in_time():
    sim = simulate(CFG)
    task = build_ev_task(CFG)
    s = sim["sessions"]
    col = SESSION_FEATURES.index("charger_faults_prev_30_nights")
    x = task.data["session"].x[:, col].numpy()
    for i in np.random.default_rng(0).choice(len(s["night"]), size=50, replace=False):
        n, c = s["night"][i], s["charger"][i]
        expected = sim["fault_log"][max(0, n - CFG.history_nights):n, c].sum()
        assert x[i] == pytest.approx(expected)


def test_labels_have_both_classes(task):
    y = torch.cat([s.y for s in task.splits.values()])
    assert 0 < float(y.mean()) < 1
