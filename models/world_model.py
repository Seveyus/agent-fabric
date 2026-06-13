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
from dataclasses import dataclass

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
    ema_tau: float = 0.99
    jepa_loss_weight: float = 1.0
    risk_loss_weight: float = 0.3   # small to avoid distorting representations
    dropout: float = 0.1


class DeliveryWorldModel(nn.Module):
    """
    JEPA world model for software delivery dynamics.

    Training objective:
        L = λ_jepa * ||s_y - ŝ_y||² + λ_risk * BCE(risk_head(s_x), labels)

    where:
        s_y   = target_encoder(graph_t_k)   [stop gradient]
        ŝ_y   = predictor(online_encoder(graph_t), events)
        s_x   = online_encoder(graph_t)

    The target encoder is updated via EMA after each optimizer step.
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
        metrics = {"jepa_loss": jepa_loss.item()}

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

    @torch.no_grad()
    def update_target(self) -> None:
        """EMA update of target encoder. Call AFTER optimizer.step()."""
        self.target_encoder.ema_update(self.online_encoder, tau=self.cfg.ema_tau)

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
