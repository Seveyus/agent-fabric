"""
What can be checked about the cuGraph-PyG backend without a GPU.

These tests exercise *our* adapter code against stand-in store classes that
record calls; they do not test cuGraph itself (see the `cugraph`-marked test in
test_backends.py for that, which needs a RAPIDS-capable GPU).
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from training import backends  # noqa: E402


class _RecordingStore(dict):
    def __setitem__(self, key, value):
        super().__setitem__(key, value)


def test_hetero_to_cugraph_stores_registers_everything(small_task, monkeypatch):
    fake = types.ModuleType("cugraph_pyg.data")
    fake.GraphStore = _RecordingStore
    fake.FeatureStore = _RecordingStore
    monkeypatch.setitem(sys.modules, "cugraph_pyg", types.ModuleType("cugraph_pyg"))
    monkeypatch.setitem(sys.modules, "cugraph_pyg.data", fake)

    data = small_task.data
    feature_store, graph_store = backends.hetero_to_cugraph_stores(data)
    # every edge type registered as COO with (num_src, num_dst) — both directions are explicit
    keys = {k[0]: k for k in graph_store}
    assert set(keys) == set(data.edge_types)
    for et, key in keys.items():
        _, layout, is_sorted, size = key
        assert layout == "coo" and is_sorted is False
        assert size == (data[et[0]].num_nodes, data[et[2]].num_nodes)
        assert torch.equal(graph_store[key], data[et].edge_index)
    # cuGraph-PyG temporal sampling needs edge times for *every* edge type
    for et in data.edge_types:
        assert torch.equal(feature_store[et, "time", None], data[et].time)
    for nt in data.node_types:
        assert torch.equal(feature_store[nt, "x", None], data[nt].x)


def test_missing_edge_time_is_rejected(small_task, monkeypatch):
    fake = types.ModuleType("cugraph_pyg.data")
    fake.GraphStore = _RecordingStore
    fake.FeatureStore = _RecordingStore
    monkeypatch.setitem(sys.modules, "cugraph_pyg", types.ModuleType("cugraph_pyg"))
    monkeypatch.setitem(sys.modules, "cugraph_pyg.data", fake)
    data = small_task.data.clone()
    del data[data.edge_types[0]].time
    with pytest.raises(ValueError, match="time"):
        backends.hetero_to_cugraph_stores(data)


def test_cugraph_backend_fails_clearly_without_gpu(small_task):
    ok, reason = backends.cugraph_available()
    if ok:
        pytest.skip("a RAPIDS-capable GPU is available here")
    with pytest.raises(RuntimeError, match="cuGraph backend unavailable"):
        backends.make_backend("cugraph", small_task.data, device="cuda")
    assert reason
