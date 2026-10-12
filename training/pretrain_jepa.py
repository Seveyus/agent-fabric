"""
Self-supervised temporal JEPA pretraining on the PR graph (no labels).

    python -m training.pretrain_jepa --config configs/synthetic_pr_merge.toml

Uses the same TOML config as training/node_task.py ([model], [sampler],
[features], [task] and the [jepa] section). Anchors are PRs opened before
``task.train_end`` (including warm-up PRs that are never labelled seeds) whose
target window t+Δ ends before ``task.val_start`` — so pretraining never sees
interactions from the validation or test periods.

The checkpoint stores the online encoder; downstream runs load it with
``[pretrain] checkpoint = ".../jepa.pt"`` (fine-tune) or additionally
``freeze_encoder = true`` (linear probe).

Backend: PyG only. The target view drops edges older than the anchor's open
time, which needs per-edge timestamps in the sampled batch; with cuGraph-PyG
the equivalent is fixed-window sampling (input_start_time / input_end_time),
which is not wired up here.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from core.dataset.temporal_graph import METADATA, TemporalTask
from models.hetero_gnn import build_encoder
from models.temporal_jepa import TemporalNodeJEPA, future_only_view
from training.backends import PyGBackend
from training.config import ExperimentConfig, load_config
from training.node_task import make_spec, prepare_task, seed_everything, select_features, standardise

log = logging.getLogger(__name__)
SEED_TYPE = "pr"


def eligible_anchors(task: TemporalTask, cfg: ExperimentConfig, author_is_bot: torch.Tensor) -> torch.Tensor:
    """PR nodes with an observed open time t such that t < train_end and t + max(Δ) <= val_start."""
    b = cfg.task.bounds()
    max_delta = int(max(cfg.jepa.delta_hours) * 3600)
    t = task.data["pr"].open_ts
    ok = (t >= 0) & (t < b["train"][1]) & (t + max_delta <= b["val"][0]) & (author_is_bot == 0)
    return torch.nonzero(ok).flatten()


def pretrain(cfg: ExperimentConfig, task: TemporalTask | None = None) -> dict:
    if cfg.backend.name != "pyg":
        raise NotImplementedError("JEPA pretraining is implemented for the PyG backend only (see module docstring)")
    if cfg.model.encoder == "mlp":
        raise ValueError("JEPA pretraining needs a message-passing encoder (sage or hgt)")
    seed_everything(cfg.experiment.seed)
    if cfg.backend.torch_threads:
        torch.set_num_threads(cfg.backend.torch_threads)

    task = task if task is not None else prepare_task(cfg)
    bot_col = task.feature_names["pr"].index("author_is_bot")
    author_is_bot = task.data["pr"].x[:, bot_col].clone()
    select_features(task, cfg.features.use_history)
    norm = standardise(task)

    anchors = eligible_anchors(task, cfg, author_is_bot)
    gen = torch.Generator().manual_seed(cfg.experiment.seed)
    if cfg.jepa.max_anchors and len(anchors) > cfg.jepa.max_anchors:
        anchors = anchors[torch.randperm(len(anchors), generator=gen)[: cfg.jepa.max_anchors]].sort().values
    if len(anchors) == 0:
        raise ValueError("no eligible JEPA anchors")
    open_ts = task.data["pr"].open_ts
    log.info("JEPA: %d anchors, Δ ∈ %s h", len(anchors), cfg.jepa.delta_hours)

    device = torch.device(cfg.backend.device)
    backend = PyGBackend(task.data, device=device, num_workers=cfg.backend.num_workers)
    spec = make_spec(cfg)
    in_dims = {nt: task.data[nt].x.size(1) for nt in task.data.node_types}
    m = cfg.model
    encoder = build_encoder(m.encoder, in_dims, METADATA, hidden=m.hidden, num_layers=m.num_layers,
                            heads=m.heads, dropout=m.dropout)
    jepa = TemporalNodeJEPA(encoder, cfg.jepa.predictor_hidden, cfg.jepa.ema_tau, cfg.jepa.ema_tau_end).to(device)
    opt = torch.optim.AdamW(jepa.trainable_parameters(), lr=cfg.jepa.lr, weight_decay=cfg.jepa.weight_decay)

    deltas = torch.tensor(cfg.jepa.delta_hours, dtype=torch.float32)
    steps_per_epoch = (len(anchors) + spec.batch_size - 1) // spec.batch_size
    total_steps = cfg.jepa.epochs * steps_per_epoch
    step = 0
    history = []
    t_start = time.perf_counter()
    for epoch in range(1, cfg.jepa.epochs + 1):
        perm = anchors[torch.randperm(len(anchors), generator=gen)]
        delta_h = deltas[torch.randint(len(deltas), (len(perm),), generator=gen)]
        t_ctx = open_ts[perm]
        t_tgt = t_ctx + (delta_h * 3600).long()
        ctx_loader = backend.loader(SEED_TYPE, perm, t_ctx, spec, shuffle=False)
        tgt_loader = backend.loader(SEED_TYPE, perm, t_tgt, spec, shuffle=False)
        jepa.train()
        agg: dict[str, list[float]] = {}
        t0 = time.perf_counter()
        for ctx, tgt in zip(ctx_loader, tgt_loader):
            ids = ctx[SEED_TYPE].input_id
            if not torch.equal(ids, tgt[SEED_TYPE].input_id):
                raise RuntimeError("context and target loaders are out of sync")
            ctx, tgt = ctx.to(device), tgt.to(device)
            future_only_view(tgt, SEED_TYPE, t_ctx[ids])
            loss, met = jepa.loss(ctx, tgt, delta_h[ids], SEED_TYPE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(jepa.trainable_parameters(), cfg.train.grad_clip)
            opt.step()
            step += 1
            met["ema_tau"] = jepa.update_target(step / max(1, total_steps))
            for k, v in met.items():
                agg.setdefault(k, []).append(v)
        rec = {"epoch": epoch, "seconds": time.perf_counter() - t0, **{k: float(np.mean(v)) for k, v in agg.items()}}
        history.append(rec)
        log.info("jepa epoch %d loss=%.4f normalised=%.3f online_std=%.3f target_std=%.3f tau=%.4f (%.0fs)",
                 epoch, rec["jepa_loss"], rec["normalised_loss"], rec["online_std"], rec["target_std"],
                 rec["ema_tau"], rec["seconds"])

    out = Path(cfg.jepa.output or Path(cfg.experiment.output_dir) / cfg.experiment.name / "jepa.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "anchors": int(len(anchors)),
        "history": history,
        "seconds": time.perf_counter() - t_start,
        "encoder_config": {"encoder": m.encoder, "hidden": m.hidden, "num_layers": m.num_layers,
                           "heads": m.heads, "dropout": m.dropout},
        "use_history": cfg.features.use_history,
    }
    torch.save({**result, "online_encoder": jepa.online.state_dict(), "normalisation": norm,
                "config": cfg.to_dict()}, out)
    (out.with_suffix(".json")).write_text(json.dumps(result, indent=2))
    log.info("saved %s", out)
    return {**result, "checkpoint": str(out)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    pretrain(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
