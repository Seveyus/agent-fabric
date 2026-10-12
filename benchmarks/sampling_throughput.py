"""
Neighbour-sampling and training-step throughput on temporal heterogeneous graphs.

    # CPU (PyG + pyg-lib), synthetic graphs of increasing size
    python -m benchmarks.sampling_throughput --backend pyg --scales 2000 20000 60000

    # the real GH Archive graph (cached task built by training/node_task.py)
    python -m benchmarks.sampling_throughput --backend pyg --task data/gharchive/task_pr_merge_48h.pt

    # GPU (cuGraph-PyG); needs a RAPIDS-capable machine — see docs/GRAPH_LEARNING.md
    python -m benchmarks.sampling_throughput --backend cugraph --device cuda --scales 2000 20000 60000

For every (graph, fanout, batch size) it reports
    sampling only    : seeds/s, sampled nodes/s and edges/s, mean subgraph size
    train step       : HGT / SAGE forward+backward on the sampled batches,
                       with time split into "wait for sampler" vs "compute"
    memory           : graph tensors (bytes), peak process RSS, CUDA peak
Results go to experiments/results/benchmark_<backend>_<host-tag>.json/.md.
Nothing is extrapolated: rows exist only for configurations that were run.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import resource
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from core.dataset.temporal_graph import METADATA, PRMergeTaskConfig, TemporalTask, build_pr_merge_task
from models.hetero_gnn import NodeClassifier, build_encoder
from training.backends import SamplerSpec, configure_pyg_kernels, make_backend

log = logging.getLogger(__name__)

SYN_TASK = PRMergeTaskConfig(train_start="2025-09-05", train_end="2025-09-11", val_start="2025-09-13",
                             val_end="2025-09-15", test_start="2025-09-17", test_end="2025-09-20")


def graph_bytes(data) -> int:
    n = 0
    for store in list(data.node_stores) + list(data.edge_stores):
        for v in store.values():
            if isinstance(v, torch.Tensor):
                n += v.numel() * v.element_size()
    return n


def synthetic_task(n_users: int) -> tuple[TemporalTask, float]:
    from core.dataset.synthetic_events import SyntheticConfig, generate_events

    t0 = time.perf_counter()
    ev = generate_events(SyntheticConfig(n_users=n_users, n_repos=max(20, n_users // 10), days=21, seed=0))
    task = build_pr_merge_task(ev, SYN_TASK)
    return task, time.perf_counter() - t0


def bench_sampling(backend, task, spec: SamplerSpec, n_batches: int, warmup: int = 2) -> dict:
    s = task.splits["train"]
    loader = backend.loader("pr", s.node, s.time, spec, shuffle=True, generator=torch.Generator().manual_seed(0))
    it = iter(loader)
    for _ in range(warmup):
        next(it, None)
    seeds = nodes = edges = batches = 0
    t0 = time.perf_counter()
    for batch in it:
        seeds += batch["pr"].batch_size
        nodes += sum(batch[nt].num_nodes for nt in batch.node_types)
        edges += sum(batch[et].edge_index.size(1) for et in batch.edge_types)
        batches += 1
        if batches >= n_batches:
            break
    dt = time.perf_counter() - t0
    return {
        "batches": batches, "seconds": dt,
        "seeds_per_s": seeds / dt, "nodes_per_s": nodes / dt, "edges_per_s": edges / dt,
        "mean_nodes_per_batch": nodes / max(1, batches), "mean_edges_per_batch": edges / max(1, batches),
        "ms_per_batch": 1000 * dt / max(1, batches),
    }


def bench_train_step(backend, task, spec: SamplerSpec, encoder: str, n_batches: int, device) -> dict:
    in_dims = {nt: task.data[nt].x.size(1) for nt in task.data.node_types}
    torch.manual_seed(0)
    model = NodeClassifier(build_encoder(encoder, in_dims, METADATA, hidden=64,
                                         num_layers=spec.num_hops, heads=2)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    s = task.splits["train"]
    it = iter(backend.loader("pr", s.node, s.time, spec, shuffle=True, generator=torch.Generator().manual_seed(1)))
    t_wait = t_compute = 0.0
    seeds = batches = 0
    for i in range(n_batches + 2):
        t0 = time.perf_counter()
        batch = next(it, None)
        if batch is None:
            break
        batch = batch.to(device)
        t1 = time.perf_counter()
        y = s.y[batch["pr"].input_id.cpu()].to(device)
        loss = F.binary_cross_entropy_with_logits(model(batch, "pr"), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i >= 2:  # warm-up
            t_wait += t1 - t0
            t_compute += t2 - t1
            seeds += batch["pr"].batch_size
            batches += 1
    total = t_wait + t_compute
    return {
        "encoder": encoder, "batches": batches,
        "seeds_per_s": seeds / max(total, 1e-9),
        "ms_per_batch": 1000 * total / max(1, batches),
        "sampling_share": t_wait / max(total, 1e-9),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="pyg", choices=["pyg", "cugraph"])
    p.add_argument("--device", default=None)
    p.add_argument("--scales", type=int, nargs="*", default=[2000, 20000],
                   help="synthetic graph sizes (number of users)")
    p.add_argument("--task", default=None, help="cached TemporalTask .pt to benchmark (e.g. the GH graph)")
    p.add_argument("--fanouts", nargs="*", default=["5,3", "10,5", "20,10"])
    p.add_argument("--batch-sizes", type=int, nargs="*", default=[256, 1024])
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--encoders", nargs="*", default=["sage", "hgt"])
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", default="experiments/results")
    p.add_argument("--tag", default=None, help="suffix for the output file names")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.threads:
        torch.set_num_threads(args.threads)
    configure_pyg_kernels("on")
    device = torch.device(args.device or ("cuda" if args.backend == "cugraph" else "cpu"))

    graphs: list[tuple[str, TemporalTask, float]] = []
    if args.task:
        t0 = time.perf_counter()
        graphs.append((f"task:{Path(args.task).name}", TemporalTask.load(args.task), time.perf_counter() - t0))
    for n in args.scales:
        task, build_s = synthetic_task(n)
        graphs.append((f"synthetic:{n}_users", task, build_s))

    rows = []
    env = None
    for name, task, build_s in graphs:
        t0 = time.perf_counter()
        backend = make_backend(args.backend, task.data, device=str(device))
        setup_s = time.perf_counter() - t0
        env = env or backend.describe()
        g = {
            "graph": name,
            "num_nodes": sum(task.data[nt].num_nodes for nt in task.data.node_types),
            "num_edges": sum(task.data[et].edge_index.size(1) for et in task.data.edge_types),
            "graph_mb": graph_bytes(task.data) / 2**20,
            "train_seeds": len(task.splits["train"]),
            "build_or_load_seconds": build_s,
            "backend_setup_seconds": setup_s,
        }
        log.info("graph %s: %d nodes, %d edges, %.0f MB", name, g["num_nodes"], g["num_edges"], g["graph_mb"])
        for fan in args.fanouts:
            fanout = [int(x) for x in fan.split(",")]
            for bs in args.batch_sizes:
                spec = SamplerSpec(fanout=None, default_fanout=fanout, batch_size=bs, temporal=True)
                row = {**g, "fanout": fanout, "batch_size": bs, "temporal": True,
                       "sampling": bench_sampling(backend, task, spec, args.batches)}
                row["train_step"] = [bench_train_step(backend, task, spec, e, max(5, args.batches // 2), device)
                                     for e in args.encoders]
                row["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                row["cuda_peak_mb"] = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None
                rows.append(row)
                sm = row["sampling"]
                log.info("  fanout=%s bs=%d: %.0f seeds/s, %.0f edges/s, %.0f nodes/batch | %s",
                         fanout, bs, sm["seeds_per_s"], sm["edges_per_s"], sm["mean_nodes_per_batch"],
                         ", ".join(f"{t['encoder']} {t['seeds_per_s']:.0f} seeds/s "
                                   f"(sampling {100 * t['sampling_share']:.0f}%)" for t in row["train_step"]))
        # non-temporal sampling for the mid fan-out, to measure the cost of time filtering
        fanout = [int(x) for x in args.fanouts[len(args.fanouts) // 2].split(",")]
        spec = SamplerSpec(fanout=None, default_fanout=fanout, batch_size=args.batch_sizes[0], temporal=False)
        rows.append({**g, "fanout": fanout, "batch_size": spec.batch_size, "temporal": False,
                     "sampling": bench_sampling(backend, task, spec, args.batches), "train_step": []})

    host = {"platform": platform.platform(), "python": platform.python_version(),
            "cpu": env.get("cpu") if env else None}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                host["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"benchmark_{args.backend}" + (f"_{args.tag}" if args.tag else "")
    (out / f"{stem}.json").write_text(json.dumps({"environment": env, "host": host, "rows": rows}, indent=2))
    lines = [f"# Sampling / training-step throughput — backend `{args.backend}`", "",
             f"- host: {host}", f"- backend: {env}", "",
             "| Graph | Nodes | Edges | Graph MB | Fanout | Batch | Temporal | Sampling seeds/s | Sampled edges/s "
             "| Nodes/batch | SAGE train seeds/s (sampling share) | HGT train seeds/s (sampling share) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ts = {t["encoder"]: f"{t['seeds_per_s']:.0f} ({100 * t['sampling_share']:.0f}%)" for t in r["train_step"]}
        sm = r["sampling"]
        lines.append(f"| {r['graph']} | {r['num_nodes']:,} | {r['num_edges']:,} | {r['graph_mb']:.0f} | "
                     f"{r['fanout']} | {r['batch_size']} | {r['temporal']} | {sm['seeds_per_s']:.0f} | "
                     f"{sm['edges_per_s']:.0f} | {sm['mean_nodes_per_batch']:.0f} | {ts.get('sage', '—')} | "
                     f"{ts.get('hgt', '—')} |")
    (out / f"{stem}.md").write_text("\n".join(lines) + "\n")
    log.info("wrote %s", out / f"{stem}.md")


if __name__ == "__main__":
    main()
