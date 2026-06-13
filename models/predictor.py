"""
JEPA Predictor — predicts the future state representation from the current
state representation conditioned on the events that occurred between T and T+k.

This is the core of the JEPA architecture:
    predictor(s_x, events) ≈ target_encoder(graph_t_k)

The predictor is trained with MSE loss in latent space.
The target encoder is NOT trained via backprop — it is updated via EMA.
This prevents representational collapse (the trivial solution where everything
maps to the same constant vector).

Reference: "Self-Supervised Learning from Images with a Joint-Embedding
Predictive Architecture" (I-JEPA) — Assran et al., 2023
https://arxiv.org/abs/2301.08243
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.dataset.graph_builder import EVENT_FEAT_DIM
from models.encoder import REPR_DIM


class DeliveryPredictor(nn.Module):
    """
    Predicts the latent representation of the project state at T+k given:
        - s_x    : representation of state at T (from online encoder)
        - events : compact event vector describing what happened between T and T+k

    Architecture: simple MLP with residual connection and layer norm.
    Kept intentionally small — the encoder does the heavy lifting.
    """

    def __init__(
        self,
        repr_dim: int = REPR_DIM,
        event_dim: int = EVENT_FEAT_DIM,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.repr_dim = repr_dim

        input_dim = repr_dim + event_dim

        layers: list[nn.Module] = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, repr_dim))
        self.net = nn.Sequential(*layers)

        # Lightweight event encoder — projects raw event vector to a richer space
        self.event_proj = nn.Sequential(
            nn.Linear(event_dim, event_dim * 2),
            nn.GELU(),
            nn.Linear(event_dim * 2, event_dim),
        )

    def forward(self, s_x: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_x    : (batch, repr_dim) — current state representation
            events : (batch, event_dim) — event summary vector

        Returns:
            s_y_hat : (batch, repr_dim) — predicted future representation
        """
        events_enc = self.event_proj(events)
        x = torch.cat([s_x, events_enc], dim=-1)
        delta = self.net(x)
        # Residual: predict the DELTA from current state, not the absolute future
        # This is more stable and aligns with the intuition that most states
        # don't change dramatically in 14 days.
        return s_x + delta


class RiskHead(nn.Module):
    """
    Lightweight supervised head on top of the encoder representation.
    Predicts delivery risk labels (regression, velocity_drop, etc.).

    Trained jointly with the JEPA loss using a small weight to avoid
    distorting the self-supervised representation.
    """

    def __init__(self, repr_dim: int = REPR_DIM, num_outputs: int = 3) -> None:
        super().__init__()
        # 3 outputs: regression, velocity_drop, release_in_t_k
        self.head = nn.Sequential(
            nn.Linear(repr_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_outputs),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (batch, num_outputs)."""
        return self.head(s)
