"""
Delivery State Encoder — heterogeneous graph encoder for project snapshots.

Architecture: Linear projection per node type → L-layer HGT (residual) →
per-node-type mean pool → concat → MLP head.

HGT (Heterogeneous Graph Transformer) keeps separate key/query/value and
message projections per node type and a relation-specific attention/message
matrix per edge type, which is why it is used here: PRs, contributors, file
zones and milestones have different feature spaces and interact through
different relations.

Reference: "Heterogeneous Graph Transformer" — Hu et al., 2020
           https://arxiv.org/abs/2003.01332

Node types and feature dims are defined in core.dataset.graph_builder. Every
forward relation has a reverse relation (rev_*) so all node types receive
messages; without them HGTConv drops node types that are never a destination.

Output: a single vector of shape (batch, REPR_DIM) representing the delivery
        state of the project during one time window.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import HGTConv, global_mean_pool
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from core.dataset.graph_builder import (  # noqa: F401  (re-exported for backward compatibility)
    EDGE_TYPES,
    METADATA,
    NODE_FEAT_DIMS,
    NODE_TYPES,
)

REPR_DIM = 128  # output representation dimension


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

        # Final projection to representation space (one pooled vector per node type)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * len(NODE_TYPES), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, repr_dim),
        )

    def forward(self, data: "HeteroData") -> torch.Tensor:
        """
        Args:
            data: PyG HeteroData (single graph or Batch) containing every node
                  type in NODE_TYPES.

        Returns:
            Tensor of shape (num_graphs, repr_dim).
        """
        missing = [nt for nt in NODE_TYPES if nt not in data.node_types]
        if missing:
            raise ValueError(f"snapshot graph is missing node types {missing}")

        x_dict = {nt: F.gelu(self.input_proj[nt](data[nt].x)) for nt in NODE_TYPES}

        # HGTConv requires edge_index_dict in metadata order (see models/hetero_gnn.py).
        edges = {et: data.edge_index_dict[et] for et in EDGE_TYPES if et in data.edge_index_dict}
        for conv in self.convs:
            out = conv(x_dict, edges)
            # Residual connection; node types without incoming edges keep their state.
            x_dict = {
                nt: self.dropout(F.gelu(out[nt] + x_dict[nt])) if out.get(nt) is not None else x_dict[nt]
                for nt in NODE_TYPES
            }

        num_graphs = int(getattr(data, "num_graphs", 1))
        pooled = []
        for nt in NODE_TYPES:
            batch = getattr(data[nt], "batch", None)
            if batch is None:
                batch = torch.zeros(x_dict[nt].size(0), dtype=torch.long, device=x_dict[nt].device)
            # Mean per node type, so large node sets (PRs, contributors) do not
            # dominate the pooled vector.
            pooled.append(self.norm(global_mean_pool(x_dict[nt], batch, size=num_graphs)))
        return self.output_proj(torch.cat(pooled, dim=-1))   # (B, repr_dim)

    @torch.no_grad()
    def ema_update(self, online: "DeliveryEncoder", tau: float = 0.99) -> None:
        """
        Update this (target) encoder as an EMA of the online encoder.
        Call after each optimizer step on the online encoder.
        """
        for p_target, p_online in zip(self.parameters(), online.parameters()):
            p_target.mul_(tau).add_(p_online.detach(), alpha=1.0 - tau)
        for b_target, b_online in zip(self.buffers(), online.buffers()):
            b_target.copy_(b_online)
