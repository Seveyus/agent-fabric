"""
JEPA training loop for the Delivery World Model.

Quick start:
    python -m training.train \
        --pairs data/pairs_14d.pkl \
        --epochs 100 \
        --batch-size 32 \
        --device cuda   # or cpu, or mps

The script saves checkpoints to checkpoints/ and logs metrics to stdout.
Optional: set WANDB_PROJECT env var to enable W&B logging.

Validation uses a temporal split (DeliveryPairDataset.temporal_split): the
validation pairs start after every training target window ends.

Model selection: the JEPA loss is not comparable across epochs (the target
encoder moves), so "best.pt" is selected on the validation risk-head loss when
the risk head is trained, otherwise only "last.pt" is written.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.world_model import DeliveryWorldModel, WorldModelConfig
from training.dataset import DeliveryPairDataset, collate_fn  # noqa: F401  (collate_fn re-exported)

log = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Delivery World Model")
    p.add_argument("--pairs", required=True, help="Path to pairs .pkl file")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--ema-tau", type=float, default=0.99)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--repr-dim", type=int, default=128)
    p.add_argument("--jepa-weight", type=float, default=1.0)
    p.add_argument("--risk-weight", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args: argparse.Namespace) -> dict:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    seed_everything(args.seed)
    device = torch.device(args.device)
    log.info("Training on device: %s", device)

    # ------------------------------------------------------------------ #
    # Dataset                                                              #
    # ------------------------------------------------------------------ #
    dataset = DeliveryPairDataset(args.pairs)
    train_ds, val_ds = dataset.temporal_split(args.val_split)
    n_train, n_val = len(train_ds), len(val_ds)
    if n_train == 0 or n_val == 0:
        raise ValueError(f"temporal split produced train={n_train} val={n_val}; collect a longer history")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=n_train > args.batch_size,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    log.info("Train: %d  Val: %d", n_train, n_val)

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    cfg = WorldModelConfig(
        hidden_dim=args.hidden_dim,
        repr_dim=args.repr_dim,
        ema_tau=args.ema_tau,
        jepa_loss_weight=args.jepa_weight,
        risk_loss_weight=args.risk_weight,
    )
    model = DeliveryWorldModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameter_groups(lr=args.lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model params: %s  (~%.1fK)", param_count, param_count / 1000)

    # Optional W&B
    wandb_run = None
    if os.environ.get("WANDB_PROJECT"):
        try:
            import wandb
            wandb_run = wandb.init(
                project=os.environ["WANDB_PROJECT"],
                config=vars(args),
            )
        except ImportError:
            log.warning("wandb not installed — skipping W&B logging")

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_risk = float("inf")
    total_steps = max(1, args.epochs * len(train_loader))
    step = 0
    history: list[dict] = []

    def _save(path: Path, epoch: int) -> None:
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "config": model.config_dict(), "args": vars(args)},
            path,
        )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_metrics: dict[str, list[float]] = {}

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss, metrics = model.training_step(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            step += 1
            metrics["ema_tau"] = model.update_target(progress=step / total_steps)  # AFTER optimizer.step()

            for k, v in metrics.items():
                train_metrics.setdefault(k, []).append(v)

        scheduler.step()

        # Validation
        model.eval()
        val_metrics: dict[str, list[float]] = {}
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                _, metrics = model.training_step(batch)
                for k, v in metrics.items():
                    val_metrics.setdefault(k, []).append(v)

        train_loss = _mean(train_metrics.get("total_loss", []))
        val_loss = _mean(val_metrics.get("total_loss", []))
        elapsed = time.time() - t0

        log.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  jepa=%.4f  "
            "repr_std=%.3f  risk_acc=%.3f  lr=%.2e  t=%.1fs",
            epoch, args.epochs, train_loss, val_loss,
            _mean(train_metrics.get("jepa_loss", [])),
            _mean(train_metrics.get("repr_std", [])),
            _mean(train_metrics.get("risk_acc", [])),
            scheduler.get_last_lr()[0],
            elapsed,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_jepa_loss": _mean(val_metrics.get("jepa_loss", [])),
            "val_risk_loss": _mean(val_metrics.get("risk_loss", [])),
            "train_repr_std": _mean(train_metrics.get("repr_std", [])),
        })

        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/jepa_loss": _mean(train_metrics.get("jepa_loss", [])),
                "train/risk_loss": _mean(train_metrics.get("risk_loss", [])),
                "train/risk_acc": _mean(train_metrics.get("risk_acc", [])),
            })

        # Save checkpoint
        if epoch % args.save_every == 0:
            path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            _save(path, epoch)
            log.info("Saved checkpoint: %s", path)

        val_risk = _mean(val_metrics.get("risk_loss", []))
        if "risk_loss" in val_metrics and val_risk < best_val_risk:
            best_val_risk = val_risk
            _save(ckpt_dir / "best.pt", epoch)

    _save(ckpt_dir / "last.pt", args.epochs)
    log.info("Training complete. Best val risk loss: %.4f", best_val_risk)
    if wandb_run:
        wandb_run.finish()
    return {"history": history, "best_val_risk_loss": best_val_risk}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    train(parse_args())
