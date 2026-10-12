"""Temporal JEPA: target-view construction, stop-gradient/EMA and anchor embargo."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from core.dataset.temporal_graph import METADATA  # noqa: E402
from models.hetero_gnn import build_encoder  # noqa: E402
from models.temporal_jepa import TemporalNodeJEPA, future_only_view  # noqa: E402
from training.backends import PyGBackend, SamplerSpec  # noqa: E402

SPEC = SamplerSpec(fanout=None, default_fanout=[6, 3], batch_size=64)


def _views(task, delta_h=48):
    s = task.splits["train"]
    b = PyGBackend(task.data)
    ctx = next(iter(b.loader("pr", s.node, s.time, SPEC, shuffle=False)))
    tgt = next(iter(b.loader("pr", s.node, s.time + int(delta_h * 3600), SPEC, shuffle=False)))
    return s, ctx, tgt


def test_future_only_view_keeps_only_future_edges_into_seeds(small_task):
    s, _, tgt = _views(small_task)
    ids = tgt["pr"].input_id
    open_t = s.time[ids]
    n_future = 0
    future_only_view(tgt, "pr", open_t)
    bs = tgt["pr"].batch_size
    assert float(tgt["pr"].x[:bs].abs().sum()) == 0.0
    for et in tgt.edge_types:
        if et[2] != "pr":
            continue
        ei, t = tgt[et].edge_index, tgt[et].time
        into = ei[1] < bs
        assert torch.all(t[into] > open_t[ei[1][into]]), f"past edge kept in {et}"
        assert torch.all(t[into] <= open_t[ei[1][into]] + 48 * 3600)
        n_future += int(into.sum())
    assert n_future > 0


def test_context_and_target_are_aligned_and_loss_trains_online_only(small_task):
    s, ctx, tgt = _views(small_task)
    assert torch.equal(ctx["pr"].input_id, tgt["pr"].input_id)
    future_only_view(tgt, "pr", s.time[tgt["pr"].input_id])
    in_dims = {nt: small_task.data[nt].x.size(1) for nt in small_task.data.node_types}
    jepa = TemporalNodeJEPA(build_encoder("sage", in_dims, METADATA, hidden=16, num_layers=2))
    loss, m = jepa.loss(ctx, tgt, torch.full((ctx["pr"].batch_size,), 48.0), "pr")
    assert torch.isfinite(loss) and m["normalised_loss"] > 0
    loss.backward()
    assert all(p.grad is None for p in jepa.target.parameters())
    assert any(p.grad is not None for p in jepa.online.parameters())
    assert any(p.grad is not None for p in jepa.predictor.parameters())


def test_ema_schedule_and_update():
    enc = torch.nn.Linear(3, 3)
    enc.out_dim = 3
    jepa = TemporalNodeJEPA(enc, ema_tau=0.9, ema_tau_end=1.0)
    assert jepa.ema_tau_at(0.0) == pytest.approx(0.9) and jepa.ema_tau_at(1.0) == pytest.approx(1.0)
    with torch.no_grad():
        jepa.online.weight.add_(1.0)
    before = jepa.target.weight.clone()
    jepa.update_target(0.0)
    assert torch.allclose(jepa.target.weight, 0.9 * before + 0.1 * jepa.online.weight)
    frozen = jepa.target.weight.clone()
    jepa.update_target(1.0)  # tau = 1 -> no change
    assert torch.allclose(jepa.target.weight, frozen)


def test_anchors_respect_embargo(small_task):
    from training.config import config_from_dict
    from training.pretrain_jepa import eligible_anchors

    cfg = config_from_dict({
        "experiment": {"name": "t"}, "data": {"source": "synthetic"},
        "task": {"train_start": "2025-09-05", "train_end": "2025-09-09", "val_start": "2025-09-11",
                 "val_end": "2025-09-12", "test_start": "2025-09-14", "test_end": "2025-09-15"},
        "jepa": {"delta_hours": [24.0, 48.0]},
    })
    bot = small_task.data["pr"].x[:, small_task.feature_names["pr"].index("author_is_bot")]
    a = eligible_anchors(small_task, cfg, bot)
    t = small_task.data["pr"].open_ts[a]
    b = cfg.task.bounds()
    assert len(a) > len(small_task.splits["train"])   # unlabelled warm-up PRs are anchors too
    assert int(t.min()) >= 0 and int(t.max()) < b["train"][1]
    assert int((t + 48 * 3600).max()) <= b["val"][0]


@pytest.mark.slow
def test_pretrain_then_finetune(tmp_path):
    from training.config import config_from_dict
    from training.node_task import run_experiment
    from training.pretrain_jepa import pretrain

    raw = {
        "experiment": {"name": "t", "output_dir": str(tmp_path)},
        "data": {"source": "synthetic", "synthetic": {"n_users": 400, "n_repos": 40, "days": 16, "seed": 5}},
        "task": {"train_start": "2025-09-05", "train_end": "2025-09-09", "val_start": "2025-09-11",
                 "val_end": "2025-09-12", "test_start": "2025-09-14", "test_end": "2025-09-15"},
        "model": {"encoder": "sage", "hidden": 16},
        "sampler": {"batch_size": 128, "default_fanout": [5, 2]},
        "train": {"epochs": 1},
        "features": {"use_history": False},
        "jepa": {"epochs": 2, "output": str(tmp_path / "j.pt")},
    }
    res = pretrain(config_from_dict(raw))
    assert len(res["history"]) == 2 and res["anchors"] > 0
    raw["pretrain"] = {"checkpoint": str(tmp_path / "j.pt"), "freeze_encoder": True}
    out = run_experiment(config_from_dict(raw))
    assert out["pretrain"]["frozen"] and out["trainable_parameters"] == 17  # linear head only
