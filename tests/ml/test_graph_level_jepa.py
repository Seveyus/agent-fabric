"""Tests for the graph-level (per-repo snapshot) JEPA pipeline in models/ + training/."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from core.dataset.graph_builder import EDGE_TYPES, NODE_TYPES, RepoTimeline  # noqa: E402
from models.world_model import DeliveryWorldModel, WorldModelConfig  # noqa: E402
from training.dataset import (  # noqa: E402
    DeliveryPairDataset,
    build_pairs_from_timelines,
    collate_fn,
    snapshot_to_hetero,
)

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
TYPES = ["pr_opened", "pr_merged", "pr_closed", "push", "issue_opened", "issue_closed", "pr_review_approved"]


def make_timeline(repo: str = "o/r", days: int = 70, seed: int = 0, revert_from_day: int | None = None) -> RepoTimeline:
    rng = random.Random(seed)
    tl = RepoTimeline(repo)
    for i in range(days * 12):
        ts = T0 + timedelta(hours=2 * i)
        day = (ts - T0).days
        title = "Revert fix" if revert_from_day is not None and day >= revert_from_day else rng.choice(["fix auth", "feat api", "test db"])
        tl.add_event({
            "repo": repo,
            "type": rng.choice(TYPES),
            "actor": f"u{rng.randint(0, 25)}",
            "timestamp": ts.isoformat(),
            "payload": {"pr_number": rng.randint(1, 40), "title": title, "additions": 10,
                        "commit_count": 2, "ref": "refs/heads/main"},
        })
    return tl


def tiny_cfg() -> WorldModelConfig:
    return WorldModelConfig(hidden_dim=16, repr_dim=8, num_heads=2, num_encoder_layers=2,
                            predictor_hidden=16, predictor_layers=2, dropout=0.0)


@pytest.fixture()
def pairs_file(tmp_path):
    return build_pairs_from_timelines([make_timeline(seed=s, repo=f"o/r{s}") for s in range(3)],
                                      tmp_path / "pairs.pkl", k_days=14, window_days=7)


def test_pairs_are_matched_by_time_offset(pairs_file):
    ds = DeliveryPairDataset(pairs_file)
    assert len(ds) > 0
    for snap_t, snap_t_k, _ in ds.samples:
        assert snap_t_k.t_start - snap_t.t_start == timedelta(days=14)


def test_gap_events_exclude_target_window(tmp_path):
    # Reverts only happen from day 21 on. For the pair (context day 7, target day 21)
    # the gap is days 14-21 and must contain no revert signal.
    tl = make_timeline(revert_from_day=21)
    path = build_pairs_from_timelines([tl], tmp_path / "p.pkl", k_days=14, window_days=7)
    ds = DeliveryPairDataset(path)
    found = False
    for snap_t, snap_t_k, gap in ds.samples:
        if snap_t.t_start == T0 + timedelta(days=7):
            found = True
            assert snap_t_k.label_is_regression == 1
            assert gap[12] == 0.0  # revert slot of the event vector
    assert found


def test_snapshot_graph_has_reverse_edges_and_valid_indices(pairs_file):
    ds = DeliveryPairDataset(pairs_file)
    g = snapshot_to_hetero(ds.samples[0][0])
    assert set(g.node_types) == set(NODE_TYPES)
    assert set(g.edge_types) == set(EDGE_TYPES)
    for (src, _, dst), ei in g.edge_index_dict.items():
        if ei.numel():
            assert int(ei[0].max()) < g[src].num_nodes
            assert int(ei[1].max()) < g[dst].num_nodes
    # not a fully-connected bipartite graph any more
    n_pr, n_zone = g["pr"].num_nodes, g["file_zone"].num_nodes
    assert g["pr", "touches", "file_zone"].edge_index.size(1) == n_pr < n_pr * n_zone


def test_training_step_gradients_and_ema(pairs_file):
    torch.manual_seed(0)
    ds = DeliveryPairDataset(pairs_file)
    batch = collate_fn([ds[i] for i in range(4)])
    model = DeliveryWorldModel(tiny_cfg())
    loss, metrics = model.training_step(batch)
    loss.backward()
    assert all(p.grad is None for p in model.target_encoder.parameters()), "target encoder must not get gradients"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.online_encoder.parameters())
    assert metrics["repr_std"] >= 0

    before = [p.clone() for p in model.target_encoder.parameters()]
    with torch.no_grad():
        for p in model.online_encoder.parameters():
            p.add_(1.0)
    tau = model.update_target(progress=0.0)
    assert tau == pytest.approx(model.cfg.ema_tau)
    moved = [(a - b).abs().sum().item() for a, b in zip(model.target_encoder.parameters(), before)]
    assert all(m > 0 for m in moved)
    assert model.ema_tau_at(1.0) == pytest.approx(model.cfg.ema_tau_end)


def test_temporal_split_has_no_overlap(pairs_file):
    ds = DeliveryPairDataset(pairs_file)
    train, val = ds.temporal_split(0.3)
    assert len(train) > 0 and len(val) > 0
    last_train_target_end = max(ds.samples[i][1].t_end for i in train.indices)
    first_val_context = min(ds.samples[i][0].t_start for i in val.indices)
    assert last_train_target_end <= first_val_context


def test_train_and_evaluate_end_to_end(pairs_file, tmp_path):
    from training.evaluate import evaluate
    from training.train import parse_args, train

    args = parse_args([
        "--pairs", str(pairs_file), "--epochs", "2", "--batch-size", "4", "--val-split", "0.3",
        "--checkpoint-dir", str(tmp_path / "ckpt"), "--hidden-dim", "16", "--repr-dim", "8",
    ])
    out = train(args)
    assert len(out["history"]) == 2
    assert (tmp_path / "ckpt" / "last.pt").exists()
    metrics = evaluate(str(tmp_path / "ckpt" / "last.pt"), str(pairs_file), split="val")
    assert metrics["n_samples"] > 0
    assert "identity_baseline_mse" in metrics
