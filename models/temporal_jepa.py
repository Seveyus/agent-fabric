"""
Temporal node-level JEPA for the PR graph.

What is predicted, and why
--------------------------
For an anchor PR p opened at time t:

    context  x = p's temporal neighbourhood at t        (edges with time <= t)
               -> exactly the view the downstream classifier gets at open time
    target   y = p's *future* interactions in (t, t+Δ]  (merge, close, reviews,
               comments, ... and who made them), with p's own features masked,
               encoded by the EMA target encoder

    s_x  = f_θ(x)[p]                      online encoder (trained by backprop)
    s_y  = LayerNorm(f_ξ(y)[p])           target encoder, ξ ← τ ξ + (1-τ) θ, stop-gradient
    ŝ_y  = g_φ(s_x, embed(log1p Δ))       predictor conditioned on the horizon Δ
    L    = || ŝ_y − s_y ||²               (mean over anchors and dimensions)

The model is asked to forecast, in latent space, what will happen to a PR in
the next Δ hours, from what is known when it is opened. No labels are used.

Why the target is "future only" (and not the full neighbourhood at t+Δ):
if the target view contained the context (p's features and its past edges),
an encoder that ignores new edges would make x and y identical and drive the
loss to zero — a temporal BYOL learns invariance to time, i.e. to exactly the
change we want to anticipate. Masking p's features and dropping edges at or
before t makes context and target carry disjoint information, as in I-JEPA
(context block vs. target blocks), with Δ playing the role of the target
position.

Collapse is monitored, not assumed away: per-dimension std of s_x / s_y and
the loss normalised by the target variance are logged (normalised loss ≈ 1
means the predictor does no better than predicting the mean target).
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def future_only_view(batch, seed_type: str, open_time: torch.Tensor):
    """
    Turn a batch sampled at time t+Δ into the JEPA target view, in place.

    * edges into a seed row with ``time <= open_time`` (the seed's past) are
      removed, so the seed only receives messages from interactions in (t, t+Δ];
    * the seed rows' own features are zeroed.

    ``open_time``: (batch_size,) open time of each seed, in input order.
    Requires edge timestamps in the batch (``store.time``), which PyG's
    temporal NeighborLoader returns.
    """
    bs = batch[seed_type].batch_size
    open_time = open_time.to(batch[seed_type].x.device)
    for et in batch.edge_types:
        if et[2] != seed_type:
            continue
        store = batch[et]
        if store.edge_index.numel() == 0:
            continue
        if not hasattr(store, "time"):
            raise ValueError("future_only_view needs edge timestamps in the sampled batch")
        dst = store.edge_index[1]
        into_seed = dst < bs
        past = torch.zeros_like(into_seed)
        past[into_seed] = store.time[into_seed] <= open_time[dst[into_seed]]
        keep = ~past
        store.edge_index = store.edge_index[:, keep]
        store.time = store.time[keep]
        if hasattr(store, "e_id"):
            store.e_id = store.e_id[keep]
    x = batch[seed_type].x.clone()
    x[:bs] = 0.0
    batch[seed_type].x = x
    return batch


class TemporalNodeJEPA(nn.Module):
    def __init__(self, encoder: nn.Module, predictor_hidden: int = 128,
                 ema_tau: float = 0.996, ema_tau_end: float = 1.0) -> None:
        super().__init__()
        d = encoder.out_dim
        self.online = encoder
        self.target = copy.deepcopy(encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.delta_embed = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.predictor = nn.Sequential(
            nn.Linear(2 * d, predictor_hidden), nn.LayerNorm(predictor_hidden), nn.GELU(),
            nn.Linear(predictor_hidden, predictor_hidden), nn.LayerNorm(predictor_hidden), nn.GELU(),
            nn.Linear(predictor_hidden, d),
        )
        self.ema_tau, self.ema_tau_end = ema_tau, ema_tau_end
        self.dim = d

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("target.")]

    def loss(self, ctx_batch, tgt_batch, delta_hours: torch.Tensor, seed_type: str) -> tuple[torch.Tensor, dict]:
        s_x = self.online(ctx_batch, seed_type)
        self.target.eval()  # deterministic targets (no dropout)
        with torch.no_grad():
            s_y = F.layer_norm(self.target(tgt_batch, seed_type), (self.dim,))
        cond = self.delta_embed(torch.log1p(delta_hours.float()).unsqueeze(-1).to(s_x.device))
        pred = self.predictor(torch.cat([s_x, cond], dim=-1))
        loss = F.mse_loss(pred, s_y)
        with torch.no_grad():
            tvar = s_y.var(dim=0).mean().clamp_min(1e-8)
            metrics = {
                "jepa_loss": float(loss),
                "normalised_loss": float(loss / tvar),     # ~1 => no better than the mean target
                "online_std": float(s_x.std(dim=0).mean()),
                "target_std": float(s_y.std(dim=0).mean()),
            }
        return loss, metrics

    def ema_tau_at(self, progress: float) -> float:
        progress = min(max(progress, 0.0), 1.0)
        return self.ema_tau_end - (self.ema_tau_end - self.ema_tau) * (math.cos(math.pi * progress) + 1) / 2

    @torch.no_grad()
    def update_target(self, progress: float) -> float:
        tau = self.ema_tau_at(progress)
        for pt, po in zip(self.target.parameters(), self.online.parameters()):
            pt.mul_(tau).add_(po.detach(), alpha=1.0 - tau)
        for bt, bo in zip(self.target.buffers(), self.online.buffers()):
            bt.copy_(bo)
        return tau
