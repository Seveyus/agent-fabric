"""
Delivery State Encoder — heterogeneous graph encoder for project snapshots.

Architecture: Linear projection per node type → 3-layer HGT → global mean pool → MLP head.

HGT (Heterogeneous Graph Transformer) handles multiple node types and edge types
natively, making it ideal for graphs where PRs, contributors, file zones, and
milestones have different feature spaces and interact differently.

Reference: "Heterogeneous Graph Transformer" — Hu et al., 2020
           https://arxiv.org/abs/2003.01332

Node types:
    pr          (PR_FEAT_DIM  = 12)
    file_zone   (FILE_FEAT_DIM = 8)
    contributor (CONTRIB_FEAT_DIM = 6)
    milestone   (MILE_FEAT_DIM = 6)

Output: a single vector of shape (batch, REPR_DIM) representing the delivery
        state of the project at time T.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lazy import to avoid hard-failing if torch-geometric is not installed
try:
    from torch_geometric.nn import HGTConv, global_mean_pool
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from core.dataset.graph_builder import (
    PR_FEAT_DIM,
    FILE_FEAT_DIM,
    CONTRIB_FEAT_DIM,
    MILE_FEAT_DIM,
)

REPR_DIM = 128  # output representation dimension

NODE_TYPES = ["pr", "file_zone", "contributor", "milestone"]
EDGE_TYPES = [
    ("pr", "touches", "file_zone"),
    ("contributor", "opened", "pr"),
    ("pr", "targets", "milestone"),
    ("contributor", "committed_to", "file_zone"),
]
NODE_FEAT_DIMS = {
    "pr": PR_FEAT_DIM,
    "file_zone": FILE_FEAT_DIM,
    "contributor": CONTRIB_FEAT_DIM,
    "milestone": MILE_FEAT_DIM,
}
METADATA = (NODE_TYPES, EDGE_TYPES)


class DeliveryEncoder(nn.Module):
    """
    Encodes a project graph snapshot into a fixed-size representation vector.

    The target encoder (used to produce the JEPA training target) is a copy
    of this module updated via EMA, never via backpropagation.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        repr_dim: int = REPR_DIM,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if not HAS_PYG:
            raise ImportError(
                "torch-geometric is required for DeliveryEncoder. "
                "Install with: pip install torch-geometric"
            )

        # Per-type input projections
        self.input_proj = nn.ModuleDict({
            node_type: nn.Linear(dim, hidden_dim)
            for node_type, dim in NODE_FEAT_DIMS.items()
        })

        # HGT message-passing layers
        self.convs = nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, METADATA, heads=num_heads)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        # Final projection to representation space
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, repr_dim),
        )

    def forward(self, data: "HeteroData") -> torch.Tensor:
        """
        Args:
            data: PyG HeteroData batch with node features x_dict and edge_index_dict.

        Returns:
            Tensor of shape (batch_size, repr_dim) — one vector per graph.
        """
        # Project each node type to hidden_dim
        x_dict = {
            ntype: F.gelu(self.input_proj[ntype](data[ntype].x))
            for ntype in NODE_TYPES
            if ntype in data.node_types
        }

        # Fill missing node types with zeros so HGT doesn't fail
        for ntype in NODE_TYPES:
            if ntype not in x_dict:
                x_dict[ntype] = torch.zeros(1, self.input_proj[list(self.input_proj.keys())[0]].out_features)

        # Graph convolution
        for conv in self.convs:
            x_dict_new = conv(x_dict, data.edge_index_dict)
            x_dict = {k: self.dropout(F.gelu(v + x_dict.get(k, 0))) for k, v in x_dict_new.items()}

        # Concatenate all node representations and pool per graph
        all_x = torch.cat([x_dict[nt] for nt in NODE_TYPES if nt in x_dict], dim=0)

        # Build batch vector mapping each node to its graph index
        batch_parts = []
        for nt in NODE_TYPES:
            if nt in data.node_types and hasattr(data[nt], "batch"):
                batch_parts.append(data[nt].batch)
        if batch_parts:
            batch = torch.cat(batch_parts, dim=0)
        else:
            batch = torch.zeros(all_x.size(0), dtype=torch.long, device=all_x.device)

        pooled = global_mean_pool(all_x, batch)          # (B, hidden_dim)
        pooled = self.norm(pooled)
        return self.output_proj(pooled)                   # (B, repr_dim)

    @torch.no_grad()
    def ema_update(self, online: "DeliveryEncoder", tau: float = 0.99) -> None:
        """
        Update this (target) encoder as an EMA of the online encoder.
        Call after each optimizer step on the online encoder.
        """
        for p_target, p_online in zip(self.parameters(), online.parameters()):
            p_target.data = tau * p_target.data + (1.0 - tau) * p_online.data
