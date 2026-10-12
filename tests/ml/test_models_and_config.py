"""Encoders, config validation and the end-to-end node classification runner."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from core.dataset.temporal_graph import METADATA  # noqa: E402
from models.hetero_gnn import NodeClassifier, build_encoder  # noqa: E402
from training.backends import PyGBackend, SamplerSpec  # noqa: E402
from training.config import ConfigError, config_from_dict, load_config  # noqa: E402

BASE = {
    "experiment": {"name": "t", "output_dir": "OUT"},
    "data": {"source": "synthetic", "synthetic": {"n_users": 500, "n_repos": 50, "days": 16, "seed": 3}},
    "task": {"train_start": "2025-09-05", "train_end": "2025-09-09", "val_start": "2025-09-11",
             "val_end": "2025-09-12", "test_start": "2025-09-14", "test_end": "2025-09-15"},
    "model": {"encoder": "sage", "hidden": 32, "num_layers": 2, "heads": 2},
    "sampler": {"batch_size": 128, "default_fanout": [6, 3]},
    "train": {"epochs": 2, "patience": 2},
}


@pytest.mark.parametrize("name", ["mlp", "sage", "hgt"])
def test_encoders_forward_backward(small_task, name):
    in_dims = {nt: small_task.data[nt].x.size(1) for nt in small_task.data.node_types}
    model = NodeClassifier(build_encoder(name, in_dims, METADATA, hidden=16, num_layers=2, heads=2))
    spec = SamplerSpec(fanout=None, default_fanout=[0] if name == "mlp" else [5, 2], batch_size=32,
                       temporal=name != "mlp")
    s = small_task.splits["train"]
    batch = next(iter(PyGBackend(small_task.data).loader("pr", s.node, s.time, spec, shuffle=False)))
    logits = model(batch, "pr")
    assert logits.shape == (32,)
    logits.sum().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_config_rejects_unknown_keys():
    raw = json.loads(json.dumps(BASE))
    raw["model"]["hiden"] = 3
    with pytest.raises(ConfigError, match="unknown keys"):
        config_from_dict(raw)


def test_config_rejects_bad_fanout_and_embargo():
    raw = json.loads(json.dumps(BASE))
    raw["sampler"]["default_fanout"] = [5]
    with pytest.raises(ConfigError, match="fanout"):
        config_from_dict(raw)
    raw = json.loads(json.dumps(BASE))
    raw["task"]["val_start"] = "2025-09-09"
    with pytest.raises(ConfigError, match="embargo"):
        config_from_dict(raw)


def test_example_configs_are_valid():
    import pathlib
    paths = sorted(pathlib.Path("configs").glob("*.toml"))
    assert paths
    for p in paths:
        load_config(p)


def test_overrides(tmp_path):
    import tomllib  # noqa: F401
    cfg = load_config("configs/synthetic_pr_merge.toml",
                      ['model.encoder="sage"', "train.epochs=3", "features.use_history=true"])
    assert cfg.model.encoder == "sage" and cfg.train.epochs == 3 and cfg.features.use_history


@pytest.mark.slow
def test_run_experiment_end_to_end(tmp_path):
    from training.node_task import run_experiment

    raw = json.loads(json.dumps(BASE))
    raw["experiment"]["output_dir"] = str(tmp_path)
    res = run_experiment(config_from_dict(raw))
    run_dir = res["run_dir"]
    for f in ("config.json", "metrics.jsonl", "best.pt", "results.json"):
        assert (tmp_path / run_dir.split(str(tmp_path) + "/")[1] / f).exists()
    assert 0.0 <= res["test"]["roc_auc"] <= 1.0
    assert res["timing"]["test_seeds_per_second"] > 0
    assert res["data_source"] == "synthetic"


@pytest.mark.slow
def test_graph_recovers_history_signal_that_mlp_cannot(tmp_path):
    """On the synthetic mechanism the merge signal lives in the author's/repo's
    history. Without history features the MLP is near chance; the temporal GNN
    must recover part of it through message passing."""
    from training.node_task import run_experiment

    aucs = {}
    for enc in ("mlp", "sage"):
        raw = json.loads(json.dumps(BASE))
        raw["experiment"]["output_dir"] = str(tmp_path)
        raw["data"]["synthetic"].update(n_users=1200, n_repos=120)
        raw["model"]["encoder"] = enc
        raw["train"].update(epochs=8, patience=3, lr=0.003)
        raw["features"] = {"use_history": False}
        aucs[enc] = run_experiment(config_from_dict(raw))["test"]["roc_auc"]
    assert aucs["mlp"] < 0.6
    assert aucs["sage"] > aucs["mlp"] + 0.05, aucs


def test_pyg_hgtconv_is_order_sensitive_and_our_wrapper_is_not(small_task):
    """Regression test for the HGTConv edge-order pitfall (see models/hetero_gnn.py)."""
    import torch_geometric.backend as pyg_backend
    from torch_geometric.nn import HGTConv

    from models.hetero_gnn import HGT

    torch.manual_seed(0)
    nts, ets = ["a", "b"], [("a", "r1", "b"), ("b", "r2", "a"), ("a", "r3", "a")]
    x = {"a": torch.randn(5, 8), "b": torch.randn(4, 8)}
    ei = {ets[0]: torch.tensor([[0, 1, 2], [0, 1, 3]]), ets[1]: torch.tensor([[0, 3], [4, 2]]),
          ets[2]: torch.tensor([[1, 2], [0, 3]])}
    reordered = {k: ei[k] for k in (ets[2], ets[0], ets[1])}
    old = pyg_backend.use_segment_matmul
    pyg_backend.use_segment_matmul = False
    try:
        conv = HGTConv(8, 8, (nts, ets), heads=2)
        raw_diff = (conv(x, ei)["b"] - conv(x, reordered)["b"]).abs().max()
        assert raw_diff > 1e-4, "upstream HGTConv became order-invariant; the workaround can be dropped"

        in_dims = {nt: small_task.data[nt].x.size(1) for nt in small_task.data.node_types}
        model = HGT(in_dims, METADATA, hidden=16, num_layers=2, heads=2, dropout=0.0).eval()
        spec = SamplerSpec(fanout=None, default_fanout=[5, 2], batch_size=32)
        s = small_task.splits["val"]
        batch = next(iter(PyGBackend(small_task.data).loader("pr", s.node, s.time, spec, shuffle=False)))
        out1 = model(batch, "pr")
        shuffled = batch.clone()
        for et in reversed(list(batch.edge_types)):   # rebuild storage in reverse order
            del shuffled[et]
        for et in reversed(list(batch.edge_types)):
            shuffled[et].edge_index = batch[et].edge_index
        assert list(shuffled.edge_types) != list(batch.edge_types)
        assert torch.allclose(out1, model(shuffled, "pr"), atol=1e-6)
    finally:
        pyg_backend.use_segment_matmul = old


@pytest.mark.slow
def test_same_seed_gives_identical_results(tmp_path):
    """Reproducibility: with backend.num_workers >= 1 (the default) two runs with
    the same seed produce identical metrics, including neighbour sampling."""
    from training.node_task import run_experiment

    raw = json.loads(json.dumps(BASE))
    raw["experiment"]["output_dir"] = str(tmp_path)
    raw["train"]["epochs"] = 1
    a = run_experiment(config_from_dict(raw))
    b = run_experiment(config_from_dict(raw))
    assert a["test"]["roc_auc"] == b["test"]["roc_auc"]
    assert a["history"][0]["train_loss"] == b["history"][0]["train_loss"]
