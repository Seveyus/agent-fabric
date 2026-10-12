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


@pytest.mark.cugraph
def test_cugraph_temporal_sampling_never_returns_future_edges(small_task):
    ok, reason = cugraph_available()
    if not ok:
        pytest.skip(reason)
    backend = make_backend("cugraph", small_task.data, device="cuda")
    _check_batches(backend, small_task)
