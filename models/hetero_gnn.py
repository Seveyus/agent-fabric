"""
Node-level encoders for sampled heterogeneous mini-batches.

All encoders share one interface so they can be compared under an identical
training loop and sampler:

    encoder(batch: HeteroData, seed_type: str) -> Tensor (batch_size, hidden)

where the first ``batch[seed_type].batch_size`` rows of ``seed_type`` are the
seed nodes (PyG and cuGraph-PyG loaders both follow this convention).

    MLPEncoder   uses only the seed node's own features (no message passing).
                 It is the non-graph control: same features, same optimiser,
                 same batches, so any gap to the GNNs is due to the graph.
    HeteroSAGE   per-relation GraphSAGE with *sum* aggregation, summed over
                 relations (R-GCN-like, cheap, strong baseline). Sum (not mean)
                 so that neighbour counts — e.g. how many of an author's past
                 PRs were merged vs closed, capped by the fan-out — survive
                 aggregation; mean aggregation only sees which relations exist.
    HGT          Heterogeneous Graph Transformer (Hu et al., 2020) via PyG's
                 HGTConv: type-specific K/Q/V projections and relation-specific
                 attention and message matrices.

Note: PyG's HGTConv does not implement HGT's relative temporal encoding (RTE).
Time enters this model only through temporal neighbour sampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, HGTConv, SAGEConv


def _seed_rows(x: torch.Tensor, batch, seed_type: str) -> torch.Tensor:
    bs = batch[seed_type].batch_size
    return x[:bs]


class _InputProjection(nn.Module):
    def __init__(self, in_dims: dict[str, int], hidden: int) -> None:
        super().__init__()
        self.lin = nn.ModuleDict({nt: nn.Linear(d, hidden) for nt, d in in_dims.items()})

    def forward(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {nt: self.lin[nt](x) for nt, x in x_dict.items() if nt in self.lin}


class MLPEncoder(nn.Module):
    def __init__(self, in_dims: dict[str, int], hidden: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, seed_type: str = "pr", **_: object) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dims[seed_type]
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden

    def forward(self, batch, seed_type: str) -> torch.Tensor:
        return self.net(_seed_rows(batch[seed_type].x, batch, seed_type))


class HeteroSAGE(nn.Module):
    def __init__(self, in_dims: dict[str, int], metadata, hidden: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, **_: object) -> None:
        super().__init__()
        node_types, edge_types = metadata
        self.inp = _InputProjection(in_dims, hidden)
        self.convs = nn.ModuleList([
            HeteroConv({et: SAGEConv((hidden, hidden), hidden, aggr="sum") for et in edge_types}, aggr="sum")
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in node_types}) for _ in range(num_layers)
        ])
        self.dropout = dropout
        self.out_dim = hidden

    def forward(self, batch, seed_type: str) -> torch.Tensor:
        h = {nt: F.gelu(v) for nt, v in self.inp(batch.x_dict).items()}
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, batch.edge_index_dict)
            h = {
                nt: F.dropout(F.gelu(norm[nt](h[nt] + out[nt])) if nt in out else h[nt],
                              p=self.dropout, training=self.training)
                for nt in h
            }
        return _seed_rows(h[seed_type], batch, seed_type)


class HGT(nn.Module):
    def __init__(self, in_dims: dict[str, int], metadata, hidden: int = 64, num_layers: int = 2,
                 heads: int = 2, dropout: float = 0.1, **_: object) -> None:
        super().__init__()
        node_types, _ = metadata
        self.inp = _InputProjection(in_dims, hidden)
        self.convs = nn.ModuleList([HGTConv(hidden, hidden, metadata, heads=heads) for _ in range(num_layers)])
        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in node_types}) for _ in range(num_layers)
        ])
        self.dropout = dropout
        self.out_dim = hidden

    def forward(self, batch, seed_type: str) -> torch.Tensor:
        h = {nt: F.gelu(v) for nt, v in self.inp(batch.x_dict).items()}
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, batch.edge_index_dict)
            # HGTConv already mixes in a learned skip connection; out[nt] is None
            # for node types that received no edges in this batch.
            h = {
                nt: F.dropout(norm[nt](out[nt]) if out.get(nt) is not None else h[nt],
                              p=self.dropout, training=self.training)
                for nt in h
            }
        return _seed_rows(h[seed_type], batch, seed_type)


ENCODERS = {"mlp": MLPEncoder, "sage": HeteroSAGE, "hgt": HGT}


def build_encoder(name: str, in_dims: dict[str, int], metadata, **kwargs) -> nn.Module:
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; choose from {sorted(ENCODERS)}")
    if name == "mlp":
        return MLPEncoder(in_dims, **kwargs)
    return ENCODERS[name](in_dims, metadata, **kwargs)


class NodeClassifier(nn.Module):
    """Encoder + linear head producing one logit per seed node."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.out_dim, 1)

    def forward(self, batch, seed_type: str) -> torch.Tensor:
        return self.head(self.encoder(batch, seed_type)).squeeze(-1)
