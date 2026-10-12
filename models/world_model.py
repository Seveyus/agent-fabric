"""
DeliveryWorldModel — full JEPA world model for software delivery.

Wraps the online encoder, target encoder (EMA), predictor, and risk head
into a single module with a unified training step.

Usage:
    model = DeliveryWorldModel()
    loss, metrics = model.training_step(batch)
    loss.backward()
    optimizer.step()
    model.update_target()  # must be called AFTER optimizer step
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import DeliveryEncoder, REPR_DIM
from models.predictor import DeliveryPredictor, RiskHead
from core.dataset.graph_builder import EVENT_FEAT_DIM

log = logging.getLogger(__name__)


@dataclass
class WorldModelConfig:
    hidden_dim: int = 256
    repr_dim: int = REPR_DIM
    num_heads: int = 4
    num_encoder_layers: int = 3
    predictor_hidden: int = 512
    predictor_layers: int = 3
    ema_tau: float = 0.99          # EMA momentum at the start of training
    ema_tau_end: float = 1.0       # cosine-annealed to this value (BYOL-style schedule)
    jepa_loss_weight: float = 1.0
    risk_loss_weight: float = 0.3   # small to avoid distorting representations
    dropout: float = 0.1


class DeliveryWorldModel(nn.Module):
    """
    JEPA world model for software delivery dynamics.

    Training objective:
        L = λ_jepa * ||s_y - ŝ_y||² + λ_risk * BCE(risk_head(s_x), labels)

    where:
        s_x   = online_encoder(graph_t)               context: window [t, t+w)
        s_y   = target_encoder(graph_t_k)             target:  window [t+k, t+k+w), stop-gradient
        ŝ_y   = predictor(s_x, events_gap)
        events_gap = event summary of [t+w, t+k), i.e. strictly between the two
                     windows. It must NOT summarise the target window itself,
                     otherwise the predictor is handed the answer.

    The target encoder is updated via EMA after each optimizer step. EMA +
    predictor + stop-gradient is the BYOL/I-JEPA recipe that empirically avoids
    collapse; it is not a guarantee, so representation_std is logged.
    """

    def __init__(self, config: WorldModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or WorldModelConfig()
        self.cfg = cfg

        self.online_encoder = DeliveryEncoder(
            hidden_dim=cfg.hidden_dim,
            repr_dim=cfg.repr_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_encoder_layers,
            dropout=cfg.dropout,
        )
        # Target encoder — deep copy, updated via EMA only
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = DeliveryPredictor(
            repr_dim=cfg.repr_dim,
            event_dim=EVENT_FEAT_DIM,
            hidden_dim=cfg.predictor_hidden,
            num_layers=cfg.predictor_layers,
            dropout=cfg.dropout,
        )
        self.risk_head = RiskHead(repr_dim=cfg.repr_dim, num_outputs=3)

    def encode(self, graph) -> torch.Tensor:
        """Encode a graph snapshot. Returns (batch, repr_dim)."""
        return self.online_encoder(graph)

    def predict(self, s_x: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        """Predict the future representation. Returns (batch, repr_dim)."""
        return self.predictor(s_x, events)

    def training_step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """
        Compute JEPA + risk head loss.

        batch keys:
            graph_t     : HeteroData at time T
            graph_t_k   : HeteroData at time T+k
            events      : Tensor (B, EVENT_FEAT_DIM)
            labels      : Tensor (B, 3) — [regression, velocity_drop, release]

        Returns:
            loss    : scalar tensor
            metrics : dict of float values for logging
        """
        graph_t = batch["graph_t"]
        graph_t_k = batch["graph_t_k"]
        events = batch["events"]
        labels = batch.get("labels")

        # Online encoder — gradients flow here
        s_x = self.online_encoder(graph_t)            # (B, repr_dim)

        # Target encoder — no gradients
        with torch.no_grad():
            s_y = self.target_encoder(graph_t_k)      # (B, repr_dim)

        # Predictor — predicts future representation
        s_y_hat = self.predictor(s_x, events)         # (B, repr_dim)

        # JEPA loss — predict in latent space
        jepa_loss = F.mse_loss(s_y_hat, s_y)

        total_loss = self.cfg.jepa_loss_weight * jepa_loss
        metrics = {
            "jepa_loss": jepa_loss.item(),
            # collapse monitors: per-dim std across the batch
            "repr_std": s_x.detach().std(dim=0).mean().item() if s_x.size(0) > 1 else float("nan"),
            "target_std": s_y.std(dim=0).mean().item() if s_y.size(0) > 1 else float("nan"),
        }

        # Optional supervised risk head
        if labels is not None and self.cfg.risk_loss_weight > 0:
            logits = self.risk_head(s_x)              # (B, 3)
            risk_loss = F.binary_cross_entropy_with_logits(
                logits, labels.float()
            )
            total_loss = total_loss + self.cfg.risk_loss_weight * risk_loss
            metrics["risk_loss"] = risk_loss.item()
            with torch.no_grad():
                preds = (logits.sigmoid() > 0.5).float()
                metrics["risk_acc"] = (preds == labels.float()).float().mean().item()

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics

    def ema_tau_at(self, progress: float) -> float:
        """Cosine schedule from ema_tau (progress=0) to ema_tau_end (progress=1)."""
        progress = min(max(progress, 0.0), 1.0)
        start, end = self.cfg.ema_tau, self.cfg.ema_tau_end
        return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0

    @torch.no_grad()
    def update_target(self, progress: float | None = None) -> float:
        """EMA update of target encoder. Call AFTER optimizer.step().

        Args:
            progress: fraction of training completed in [0, 1]; None keeps tau fixed.
        Returns:
            the tau that was used.
        """
        tau = self.cfg.ema_tau if progress is None else self.ema_tau_at(progress)
        self.target_encoder.ema_update(self.online_encoder, tau=tau)
        return tau

    def config_dict(self) -> dict:
        return asdict(self.cfg)

    def parameter_groups(self, lr: float = 1e-4, predictor_lr_factor: float = 2.0) -> list[dict]:
        """
        Return optimizer parameter groups.
        The predictor uses a higher LR because it is trained more aggressively.
        """
        return [
            {"params": self.online_encoder.parameters(), "lr": lr},
            {"params": self.predictor.parameters(), "lr": lr * predictor_lr_factor},
            {"params": self.risk_head.parameters(), "lr": lr},
        ]
