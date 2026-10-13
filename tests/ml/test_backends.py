"""Sampler contract tests: seed ordering and *no future edges* (temporal leakage)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from training.backends import PyGBackend, SamplerSpec, cugraph_available, make_backend  # noqa: E402

SPEC = SamplerSpec(fanout=None, default_fanout=[6, 3], batch_size=64, temporal=True)


def _check_batches(backend, task, n_batches=5):
    s = task.splits["val"]
    loader = backend.loader("pr", s.node, s.time, SPEC, shuffle=True, generator=torch.Generator().manual_seed(0))
    seen = 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        bs = batch["pr"].batch_size
        input_id = batch["pr"].input_id.cpu()
        # seeds come first, in input order
        assert torch.equal(batch["pr"].n_id[:bs].cpu(), s.node[input_id])
        seed_time = s.time[input_id]
        n_edges = 0
        for (src, rel, dst) in batch.edge_types:
            store = batch[src, rel, dst]
            if store.edge_index.numel() == 0:
                continue
            # disjoint sampling: every node belongs to exactly one seed's subgraph
            owner = batch[dst].batch[store.edge_index[1]].cpu()
            assert torch.all(store.time.cpu() <= seed_time[owner]), f"future edge sampled in {rel}"
            n_edges += store.edge_index.size(1)
        assert n_edges > 0
        seen += bs
    assert seen > 0


def test_pyg_temporal_sampling_never_returns_future_edges(small_task):
    _check_batches(PyGBackend(small_task.data), small_task)


def test_seed_own_outcome_edges_are_invisible(small_task):
    """No sampled outcome edge (merge / close / authored_*, either direction) may
    involve the seed PR itself: those edges are timestamped after it was opened."""
    s = small_task.splits["test"]
    loader = PyGBackend(small_task.data).loader("pr", s.node, s.time, SPEC, shuffle=False)
    outcome = {"merged_into", "closed_unmerged_in", "merges", "closes", "authored_merged", "authored_closed"}
    n_outcome_edges = 0
    for batch in loader:
        seed_global = s.node[batch["pr"].input_id.cpu()]
        for (src, rel, dst) in batch.edge_types:
            if rel.removeprefix("rev_") not in outcome:
                continue
            ei = batch[src, rel, dst].edge_index
            if ei.numel() == 0:
                continue
            pr_local = ei[0] if src == "pr" else ei[1]
            owner = batch["pr"].batch[pr_local]
            same = batch["pr"].n_id[pr_local] == seed_global[owner]
            assert int(same.sum()) == 0, f"{rel}: a seed sees its own outcome"
            n_outcome_edges += ei.size(1)
    assert n_outcome_edges > 0  # outcome edges of *older* PRs are visible


def test_non_temporal_spec_requires_no_times(small_task):
    spec = SamplerSpec(fanout=None, default_fanout=[0], batch_size=32, temporal=False)
    s = small_task.splits["train"]
    batch = next(iter(PyGBackend(small_task.data).loader("pr", s.node, None, spec, shuffle=False)))
    assert batch["pr"].batch_size == 32


def test_fanout_validation():
    spec = SamplerSpec(fanout={("a", "r", "b"): [1]}, default_fanout=[2, 2])
    with pytest.raises(ValueError):
        spec.num_neighbors([("a", "r", "b")])


def test_unknown_backend(small_task):
    with pytest.raises(ValueError):
        make_backend("nope", small_task.data)


def _cugraph_violations(backend, task, fanout: list[int], n_seeds: int = 30) -> dict:
    """One seed per batch, so every sampled edge belongs to that seed. cuGraph-PyG
    heterogeneous batches carry e_id but no edge times, so times are looked up
    in the original graph through e_id."""
    data = task.data
    s = task.splits["val"]
    spec = SamplerSpec(fanout=None, default_fanout=fanout, batch_size=1, temporal=True)
    out = {"edges": 0, "future": 0, "e_id_mismatch": 0}
    for k in range(min(n_seeds, len(s))):
        batch = next(iter(backend.loader("pr", s.node[k:k + 1], s.time[k:k + 1], spec, shuffle=False)))
        assert int(batch["pr"].n_id[0]) == int(s.node[k])
        for et in batch.edge_types:
            st = batch[et]
            if st.edge_index.numel() == 0:
                continue
            e_id = st.e_id.cpu()
            src = batch[et[0]].n_id.cpu()[st.edge_index[0].cpu()]
            dst = batch[et[2]].n_id.cpu()[st.edge_index[1].cpu()]
            out["edges"] += e_id.numel()
            out["future"] += int((data[et].time[e_id] > s.time[k]).sum())
            out["e_id_mismatch"] += int(((data[et].edge_index[0, e_id] != src) |
                                         (data[et].edge_index[1, e_id] != dst)).sum())
    return out


@pytest.fixture(scope="module")
def cugraph_backend(small_task):
    ok, reason = cugraph_available()
    if not ok:
        pytest.skip(reason)
    return make_backend("cugraph", small_task.data, device="cuda")


@pytest.mark.cugraph
def test_cugraph_shuffled_batches_have_aligned_seeds(cugraph_backend, small_task):
    s = small_task.splits["val"]
    for i, batch in enumerate(cugraph_backend.loader("pr", s.node, s.time, SPEC, shuffle=True)):
        if i >= 5:
            break
        bs = batch["pr"].batch_size
        ids = batch["pr"].input_id.cpu()
        assert torch.equal(batch["pr"].n_id[:bs].cpu(), s.node[ids])
        assert torch.equal(batch["pr"].seed_time.cpu(), s.time[ids])


@pytest.mark.cugraph
def test_cugraph_one_hop_temporal_sampling_is_correct(cugraph_backend, small_task):
    r = _cugraph_violations(cugraph_backend, small_task, [6])
    assert r["edges"] > 0 and r["future"] == 0 and r["e_id_mismatch"] == 0, r


@pytest.mark.cugraph
@pytest.mark.xfail(strict=True, reason=(
    "cugraph-pyg 26.10, measured on a GTX 1650: 2-hop heterogeneous temporal sampling "
    "returns ~0.6-0.9% edges newer than the seed and ~0.7% edges whose e_id disagrees "
    "with the reported endpoints (scripts/cugraph_temporal_check.py, "
    "experiments/results/cugraph_temporal_check.json). Remove the xfail once fixed upstream."))
def test_cugraph_two_hop_temporal_sampling_never_returns_future_edges(cugraph_backend, small_task):
    r = _cugraph_violations(cugraph_backend, small_task, [6, 3])
    assert r["edges"] > 0 and r["future"] == 0 and r["e_id_mismatch"] == 0, r
