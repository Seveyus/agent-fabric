"""
Check cuGraph-PyG temporal heterogeneous neighbour sampling against the graph.

    PYTHONPATH=. .venv-gpu/bin/python scripts/cugraph_temporal_check.py [--seeds 60]

For one-seed batches (so every sampled edge belongs to that seed), every
sampled edge is checked in two independent ways:

  * e_id consistency: the edge designated by ``e_id`` in the original graph
    has the endpoints reported by ``edge_index`` + ``n_id``;
  * temporal rule: the earliest original edge between the reported endpoints
    is not after the seed time (time-respecting sampling must never return a
    pair that only exists in the future), and the edge designated by e_id is
    not after the seed time.

Run for several sampler settings to separate configuration issues from
sampler behaviour. Requires the GPU environment (scripts/setup_gpu_env.sh).
"""

from __future__ import annotations

import argparse
import collections
import json

import torch

from core.dataset.synthetic_events import SyntheticConfig, generate_events
from core.dataset.temporal_graph import PRMergeTaskConfig, build_pr_merge_task
from training.backends import CuGraphBackend


def earliest_times(data, edge_types):
    out = {}
    for et in edge_types:
        d: dict[tuple[int, int], int] = {}
        for a, c, t in zip(*data[et].edge_index.tolist(), data[et].time.tolist()):
            if t < d.get((a, c), 1 << 62):
                d[(a, c)] = t
        out[et] = d
    return out


def check(backend, task, n_seeds: int, fanout: list[int], comparison: str, disjoint: bool) -> dict:
    from cugraph_pyg.loader import NeighborLoader

    data = task.data
    s = task.splits["val"]
    mint = earliest_times(data, backend.edge_types)
    num_neighbors = {et: list(fanout) for et in backend.edge_types}
    c = collections.Counter()
    by_rel = collections.Counter()
    for k in range(min(n_seeds, len(s))):
        loader = NeighborLoader(backend.stores, num_neighbors=num_neighbors,
                                input_nodes=("pr", s.node[k:k + 1]), input_time=s.time[k:k + 1],
                                time_attr="time", temporal_strategy="uniform",
                                temporal_comparison=comparison, disjoint=disjoint, batch_size=1)
        batch = next(iter(loader))
        t0 = int(s.time[k])
        assert int(batch["pr"].n_id[0]) == int(s.node[k])
        for et in batch.edge_types:
            st = batch[et]
            if st.edge_index.numel() == 0:
                continue
            src = batch[et[0]].n_id.cpu()[st.edge_index[0].cpu()].tolist()
            dst = batch[et[2]].n_id.cpu()[st.edge_index[1].cpu()].tolist()
            for j, e in enumerate(st.e_id.cpu().tolist()):
                c["edges"] += 1
                ep = (int(data[et].edge_index[0, e]), int(data[et].edge_index[1, e]))
                if ep != (src[j], dst[j]):
                    c["e_id_endpoint_mismatch"] += 1
                if mint[et].get((src[j], dst[j]), 1 << 62) > t0:
                    c["reported_pair_only_in_future"] += 1
                    by_rel[et[1]] += 1
                if int(data[et].time[e]) > t0:
                    c["e_id_edge_after_seed_time"] += 1
    return {"comparison": comparison, "disjoint": disjoint, "fanout": fanout, "seeds": min(n_seeds, len(s)),
            **c, "future_pairs_by_relation": dict(by_rel)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=60)
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    ev = generate_events(SyntheticConfig(n_users=600, n_repos=60, days=16, seed=2))
    task = build_pr_merge_task(ev, PRMergeTaskConfig(
        train_start="2025-09-05", train_end="2025-09-09", val_start="2025-09-11",
        val_end="2025-09-12", test_start="2025-09-14", test_end="2025-09-15"))
    backend = CuGraphBackend(task.data, device="cuda")
    rows = []
    for comparison in ("monotonically_decreasing", "strictly_decreasing"):
        for disjoint in (True, False):
            for fanout in ([6, 3], [6]):
                try:
                    r = check(backend, task, args.seeds, fanout, comparison, disjoint)
                except Exception as exc:  # record sampler failures instead of stopping
                    r = {"comparison": comparison, "disjoint": disjoint, "fanout": fanout,
                         "error": f"{type(exc).__name__}: {exc}"}
                rows.append(r)
                print(json.dumps(r))
    if args.out:
        import cugraph_pyg

        with open(args.out, "w") as f:
            json.dump({"cugraph_pyg": getattr(cugraph_pyg, "__version__", "?"), "torch": torch.__version__,
                       "gpu": torch.cuda.get_device_name(), "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
