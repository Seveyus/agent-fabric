"""
Config-driven node classification on the temporal PR graph.

    python -m training.node_task --config configs/synthetic_pr_merge.toml
    python -m training.node_task --config configs/gh_pr_merge.toml --set model.encoder="sage"

One run = build/load the task, standardise features with training-seed
statistics, train with early stopping on validation ROC-AUC, reload the best
checkpoint, pick the decision threshold on validation, and report validation
and test metrics plus timing (sampling vs compute) and memory.

Outputs (in <output_dir>/<name>/<timestamp>/):
    config.json     resolved configuration
    metrics.jsonl   one line per epoch
    best.pt         best checkpoint (model, config, feature normalisation)
    results.json    final metrics, timings, environment
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import resource
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.dataset.temporal_graph import (
    HISTORY_FEATURES,
    METADATA,
    TemporalTask,
    build_pr_merge_task,
    load_events,
)
from models.hetero_gnn import NodeClassifier, build_encoder
from training.backends import SamplerSpec, make_backend
from training.config import ExperimentConfig, load_config
from training.metrics import best_f1_threshold, binary_metrics, bootstrap_auc_ci

log = logging.getLogger(__name__)
SEED_TYPE = "pr"


# --------------------------------------------------------------------------- #
# Data preparation                                                            #
# --------------------------------------------------------------------------- #

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_task(cfg: ExperimentConfig) -> TemporalTask:
    """Load the cached task if it matches the task config, otherwise build it."""
    cache = Path(cfg.data.cache) if cfg.data.cache else None
    if cache is not None and cache.exists():
        task = TemporalTask.load(cache)
        if task.meta.get("config") == asdict(cfg.task):
            log.info("Loaded cached task %s", cache)
            return task
        log.warning("Cached task %s was built with a different task config; rebuilding", cache)

    t0 = time.perf_counter()
    if cfg.data.source == "synthetic":
        from core.dataset.synthetic_events import SyntheticConfig, generate_events

        s = cfg.data.synthetic
        events = generate_events(SyntheticConfig(n_users=s.n_users, n_repos=s.n_repos, days=s.days, seed=s.seed))
    else:
        events = load_events(cfg.data.events, until_ts=cfg.task.data_end)
    task = build_pr_merge_task(events, cfg.task)
    task.meta["build_seconds"] = time.perf_counter() - t0
    task.meta["data_source"] = cfg.data.source
    if cache is not None:
        task.save(cache)
        log.info("Cached task to %s", cache)
    return task


def select_features(task: TemporalTask, use_history: bool) -> list[str]:
    """Optionally drop the history aggregates from *all* PR nodes."""
    names = list(task.feature_names["pr"])
    if use_history:
        return names
    keep = [i for i, n in enumerate(names) if n not in HISTORY_FEATURES]
    task.data["pr"].x = task.data["pr"].x[:, keep].contiguous()
    names = [names[i] for i in keep]
    task.feature_names["pr"] = names
    return names


def standardise(task: TemporalTask) -> dict[str, list[float]]:
    """z-score PR features with statistics of the *training seeds* only."""
    x = task.data["pr"].x
    ref = x[task.splits["train"].node]
    mean, std = ref.mean(0), ref.std(0).clamp_min(1e-6)
    task.data["pr"].x = (x - mean) / std
    return {"mean": mean.tolist(), "std": std.tolist()}


def subsample_train(task: TemporalTask, fraction: float, seed: int) -> None:
    """Keep a deterministic fraction of training seeds (feature statistics are
    computed before subsampling, so they do not change with the fraction)."""
    s = task.splits["train"]
    n = max(1, int(round(len(s) * fraction)))
    g = torch.Generator().manual_seed(seed)
    keep = torch.randperm(len(s), generator=g)[:n].sort().values
    task.splits["train"] = type(s)(node=s.node[keep], time=s.time[keep], y=s.y[keep])


def make_spec(cfg: ExperimentConfig) -> SamplerSpec:
    if cfg.model.encoder == "mlp":  # no message passing: sample the seeds only
        return SamplerSpec(fanout=None, default_fanout=[0], batch_size=cfg.sampler.batch_size, temporal=False)
    fanout = {tuple(k.split("/")): v for k, v in cfg.sampler.fanout.items()}
    unknown = set(fanout) - set(METADATA[1])
    if unknown:
        raise ValueError(f"sampler.fanout refers to unknown edge types: {sorted(unknown)}")
    return SamplerSpec(fanout=fanout, default_fanout=cfg.sampler.default_fanout,
                       batch_size=cfg.sampler.batch_size, temporal=cfg.sampler.temporal)


def build_model(cfg: ExperimentConfig, in_dims: dict[str, int]) -> NodeClassifier:
    m = cfg.model
    enc = build_encoder(m.encoder, in_dims, METADATA, hidden=m.hidden, num_layers=m.num_layers,
                        heads=m.heads, dropout=m.dropout, seed_type=SEED_TYPE)
    return NodeClassifier(enc)


def load_pretrained_encoder(model: NodeClassifier, cfg: ExperimentConfig) -> dict:
    blob = torch.load(cfg.pretrain.checkpoint, map_location="cpu", weights_only=False)
    enc_cfg = blob["encoder_config"]
    for k in ("encoder", "hidden", "num_layers", "heads"):
        if enc_cfg[k] != getattr(cfg.model, k):
            raise ValueError(f"pretrained encoder has {k}={enc_cfg[k]}, config has {getattr(cfg.model, k)}")
    if blob.get("use_history") != cfg.features.use_history:
        raise ValueError("pretrained encoder was trained with a different features.use_history")
    model.encoder.load_state_dict(blob["online_encoder"])
    if cfg.pretrain.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    return {"checkpoint": cfg.pretrain.checkpoint, "frozen": cfg.pretrain.freeze_encoder}


# --------------------------------------------------------------------------- #
# Train / eval loops                                                          #
# --------------------------------------------------------------------------- #

def _run(model, loader, labels: torch.Tensor, device, optimizer=None, max_batches: int = 0,
         grad_clip: float = 1.0) -> dict:
    train = optimizer is not None
    model.train(train)
    ys, ps, losses = [], [], []
    t_sample = t_compute = 0.0
    n_batches = n_nodes = n_edges = 0
    it = iter(loader)
    while True:
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = batch.to(device)
        t1 = time.perf_counter()
        y = labels[batch[SEED_TYPE].input_id.cpu()].to(device)
        with torch.set_grad_enabled(train):
            logits = model(batch, SEED_TYPE)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_sample += t1 - t0
        t_compute += t2 - t1
        losses.append(float(loss.detach()))
        ys.append(y.detach().cpu())
        ps.append(torch.sigmoid(logits.detach()).cpu())
        n_batches += 1
        n_nodes += sum(batch[nt].num_nodes for nt in batch.node_types)
        n_edges += sum(batch[et].edge_index.size(1) for et in batch.edge_types)
        if max_batches and n_batches >= max_batches:
            break
    y = torch.cat(ys).numpy() if ys else np.zeros(0)
    p = torch.cat(ps).numpy() if ps else np.zeros(0)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "y": y, "p": p,
        "sample_s": t_sample, "compute_s": t_compute, "batches": n_batches,
        "seeds": int(len(y)),
        "mean_nodes_per_batch": n_nodes / max(1, n_batches),
        "mean_edges_per_batch": n_edges / max(1, n_batches),
    }


def _environment(backend) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {**backend.describe(), "git_commit": commit, "git_dirty": dirty, "cpu": cpu,
            "cuda_available": torch.cuda.is_available()}


def run_experiment(cfg: ExperimentConfig, task: TemporalTask | None = None) -> dict:
    seed_everything(cfg.experiment.seed)
    if cfg.backend.torch_threads:
        torch.set_num_threads(cfg.backend.torch_threads)
    run_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    task = task if task is not None else prepare_task(cfg)
    feature_names = select_features(task, cfg.features.use_history)
    norm = standardise(task)
    if cfg.train.label_fraction < 1.0:
        subsample_train(task, cfg.train.label_fraction, cfg.experiment.seed)
    for name, s in task.splits.items():
        if len(s) == 0:
            raise ValueError(f"split {name!r} is empty — check the task time ranges")
        log.info("%s: %d seeds, positive rate %.3f", name, len(s), float(s.y.mean()))

    device = torch.device(cfg.backend.device)
    backend = make_backend(cfg.backend.name, task.data, device=cfg.backend.device,
                           **({"num_workers": cfg.backend.num_workers} if cfg.backend.name == "pyg" else {}))
    spec = make_spec(cfg)
    in_dims = {nt: task.data[nt].x.size(1) for nt in task.data.node_types}
    model = build_model(cfg, in_dims).to(device)
    pretrain_info = None
    if cfg.pretrain.checkpoint:
        pretrain_info = load_pretrained_encoder(model, cfg)
    elif cfg.pretrain.freeze_random_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        pretrain_info = {"checkpoint": None, "frozen": True, "random_init": True}
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model %s: %d trainable parameters", cfg.model.encoder, n_params)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    gen = torch.Generator().manual_seed(cfg.experiment.seed)

    def loader(split: str, shuffle: bool):
        s = task.splits[split]
        return backend.loader(SEED_TYPE, s.node, s.time, spec, shuffle=shuffle, generator=gen if shuffle else None)

    best_auc, best_epoch, bad = -1.0, 0, 0
    history = []
    train_seconds = 0.0
    with open(run_dir / "metrics.jsonl", "w") as mf:
        for epoch in range(1, cfg.train.epochs + 1):
            t0 = time.perf_counter()
            tr = _run(model, loader("train", True), task.splits["train"].y, device, opt,
                      cfg.train.max_batches_per_epoch, cfg.train.grad_clip)
            train_seconds += time.perf_counter() - t0
            va = _run(model, loader("val", False), task.splits["val"].y, device)
            val_m = binary_metrics(va["y"], va["p"], 0.5)
            rec = {
                "epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"],
                "val_roc_auc": val_m["roc_auc"], "val_avg_precision": val_m["avg_precision"],
                "epoch_seconds": time.perf_counter() - t0,
                "train_sample_s": tr["sample_s"], "train_compute_s": tr["compute_s"],
            }
            history.append(rec)
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            log.info("epoch %d train_loss=%.4f val_loss=%.4f val_auc=%.4f (sample %.1fs / compute %.1fs)",
                     epoch, tr["loss"], va["loss"], val_m["roc_auc"], tr["sample_s"], tr["compute_s"])
            if val_m["roc_auc"] > best_auc:
                best_auc, best_epoch, bad = val_m["roc_auc"], epoch, 0
                torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "feature_names": feature_names,
                            "normalisation": norm, "epoch": epoch}, run_dir / "best.pt")
            else:
                bad += 1
                if bad >= cfg.train.patience:
                    log.info("early stopping at epoch %d (best %d)", epoch, best_epoch)
                    break

    model.load_state_dict(torch.load(run_dir / "best.pt", weights_only=False)["model"])
    va = _run(model, loader("val", False), task.splits["val"].y, device)
    te = _run(model, loader("test", False), task.splits["test"].y, device)
    thr = best_f1_threshold(va["y"], va["p"])
    test_m = binary_metrics(te["y"], te["p"], thr)
    test_m["roc_auc_ci95"] = bootstrap_auc_ci(te["y"], te["p"], seed=cfg.experiment.seed)
    results = {
        "name": cfg.experiment.name,
        "model": cfg.model.encoder,
        "use_history": cfg.features.use_history,
        "label_fraction": cfg.train.label_fraction,
        "data_source": task.meta.get("data_source", cfg.data.source),
        "trainable_parameters": n_params,
        "pretrain": pretrain_info,
        "best_epoch": best_epoch,
        "val": binary_metrics(va["y"], va["p"], thr),
        "test": test_m,
        "timing": {
            "train_seconds_total": train_seconds,
            "epochs_run": len(history),
            "train_sample_seconds_last_epoch": history[-1]["train_sample_s"],
            "train_compute_seconds_last_epoch": history[-1]["train_compute_s"],
            "test_inference_seconds": te["sample_s"] + te["compute_s"],
            "test_sample_seconds": te["sample_s"],
            "test_forward_seconds": te["compute_s"],
            "test_seeds_per_second": te["seeds"] / max(1e-9, te["sample_s"] + te["compute_s"]),
            "test_ms_per_batch": 1000 * (te["sample_s"] + te["compute_s"]) / max(1, te["batches"]),
            "batch_size": cfg.sampler.batch_size,
            "mean_nodes_per_batch": te["mean_nodes_per_batch"],
            "mean_edges_per_batch": te["mean_edges_per_batch"],
        },
        "memory": {
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "cuda_max_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
        },
        "task": {k: task.meta.get(k) for k in ("num_nodes", "split_sizes", "split_pos_rate", "num_events")},
        "num_edges_total": sum(task.meta.get("num_edges", {}).values()),
        "environment": _environment(backend),
        "history": history,
        "run_dir": str(run_dir),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    log.info("TEST %s: roc_auc=%.4f ap=%.4f f1=%.4f (threshold %.3f from val)",
             cfg.model.encoder, test_m["roc_auc"], test_m["avg_precision"], test_m["f1"], thr)
    return results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help='override, e.g. --set model.encoder="sage" (repeatable)')
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(args.config, args.overrides)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
