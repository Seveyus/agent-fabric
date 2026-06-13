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
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from models.world_model import DeliveryWorldModel, WorldModelConfig
from training.dataset import DeliveryPairDataset

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
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
    return p.parse_args()


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate: PyG handles graph batching natively.
    Events and labels are standard tensors.
    """
    try:
        from torch_geometric.data import Batch
    except ImportError:
        raise ImportError("torch-geometric required for training")

    return {
        "graph_t": Batch.from_data_list([item["graph_t"] for item in batch]),
        "graph_t_k": Batch.from_data_list([item["graph_t_k"] for item in batch]),
        "events": torch.stack([item["events"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    device = torch.device(args.device)
    log.info("Training on device: %s", device)

    # ------------------------------------------------------------------ #
    # Dataset                                                              #
    # ------------------------------------------------------------------ #
    dataset = DeliveryPairDataset(args.pairs)
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
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
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_metrics: dict[str, list[float]] = {}

        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v.to(device)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            loss, metrics = model.training_step(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_target()   # EMA update — must be AFTER optimizer.step()

            for k, v in metrics.items():
                train_metrics.setdefault(k, []).append(v)

        scheduler.step()

        # Validation
        model.eval()
        val_metrics: dict[str, list[float]] = {}
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                _, metrics = model.training_step(batch)
                for k, v in metrics.items():
                    val_metrics.setdefault(k, []).append(v)

        train_loss = _mean(train_metrics.get("total_loss", []))
        val_loss = _mean(val_metrics.get("total_loss", []))
        elapsed = time.time() - t0

        log.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  jepa=%.4f  "
            "risk_acc=%.3f  lr=%.2e  t=%.1fs",
            epoch, args.epochs, train_loss, val_loss,
            _mean(train_metrics.get("jepa_loss", [])),
            _mean(train_metrics.get("risk_acc", [])),
            scheduler.get_last_lr()[0],
            elapsed,
        )

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
            torch.save({"epoch": epoch, "model": model.state_dict()}, path)
            log.info("Saved checkpoint: %s", path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_dir / "best.pt")

    log.info("Training complete. Best val loss: %.4f", best_val_loss)
    if wandb_run:
        wandb_run.finish()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    train(parse_args())
