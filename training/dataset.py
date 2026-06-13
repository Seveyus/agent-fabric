"""
PyTorch Dataset for JEPA training pairs.

Each sample is a (T, T+k) snapshot pair from the same repo.
The DataLoader yields batches of:
    graph_t     HeteroData   — project state at time T
    graph_t_k   HeteroData   — project state at time T+k (the target)
    events      Tensor       — (EVENT_FEAT_DIM,) event summary T→T+k
    labels      Tensor       — (3,) [regression, velocity_drop, release_in_t_k]

The dataset is built offline from GH Archive data using core/dataset/.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from core.dataset.graph_builder import (
    Snapshot,
    NODE_TYPES,
    NODE_FEAT_DIMS,
    EDGE_TYPES,
)
from core.dataset.labeler import label_pair

log = logging.getLogger(__name__)


class DeliveryPairDataset(Dataset):
    """
    Dataset of (snapshot_T, snapshot_T+k) pairs with labels.

    Args:
        pairs_path: path to a pickle file containing list of (Snapshot, Snapshot)
                    or list of dicts with keys "snap_t", "snap_t_k".
        k_days:     prediction horizon (used for documentation only; the pairs
                    are pre-computed so this is informational).
    """

    def __init__(self, pairs_path: str | Path, k_days: int = 14) -> None:
        self.k_days = k_days
        pairs_path = Path(pairs_path)
        log.info("Loading pairs from %s", pairs_path)
        with open(pairs_path, "rb") as f:
            raw = pickle.load(f)

        self.samples: list[tuple[Snapshot, Snapshot]] = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                self.samples.append((item[0], item[1]))
            elif isinstance(item, dict):
                self.samples.append((item["snap_t"], item["snap_t_k"]))

        log.info("Loaded %d training pairs", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        snap_t, snap_t_k = self.samples[idx]
        graph_t = _snapshot_to_hetero(snap_t)
        graph_t_k = _snapshot_to_hetero(snap_t_k)
        events = torch.tensor(snap_t_k.to_event_vector(), dtype=torch.float32)
        lb = label_pair(snap_t, snap_t_k)
        labels = torch.tensor(
            [lb["regression"], lb["velocity_drop"], lb["release_in_t_k"]],
            dtype=torch.float32,
        )
        return {
            "graph_t": graph_t,
            "graph_t_k": graph_t_k,
            "events": events,
            "labels": labels,
            "repo": snap_t.repo,
        }


def _snapshot_to_hetero(snap: Snapshot) -> "HeteroData":
    """Convert a Snapshot into a PyG HeteroData object."""
    if not HAS_PYG:
        raise ImportError("torch-geometric required")

    data = HeteroData()
    feats = snap.to_node_features()

    for ntype in NODE_TYPES:
        if ntype in feats:
            x = torch.tensor(feats[ntype], dtype=torch.float32)
        else:
            x = torch.zeros(1, NODE_FEAT_DIMS[ntype], dtype=torch.float32)
        data[ntype].x = x

    # Build simple edge indices based on node counts
    # In absence of explicit file paths, we connect all PRs to all file_zones
    # and all contributors to all PRs (fully-connected within type pairs).
    # This is a conservative approximation — richer edges need actual file data.
    _add_full_edges(data, "pr", "touches", "file_zone")
    _add_full_edges(data, "contributor", "opened", "pr")
    _add_full_edges(data, "pr", "targets", "milestone")
    _add_full_edges(data, "contributor", "committed_to", "file_zone")

    return data


def _add_full_edges(data: "HeteroData", src: str, rel: str, dst: str) -> None:
    """Add fully-connected bipartite edges between src and dst node types."""
    n_src = data[src].x.size(0)
    n_dst = data[dst].x.size(0)
    if n_src == 0 or n_dst == 0:
        data[src, rel, dst].edge_index = torch.zeros((2, 0), dtype=torch.long)
        return
    src_idx = torch.arange(n_src).repeat_interleave(n_dst)
    dst_idx = torch.arange(n_dst).repeat(n_src)
    data[src, rel, dst].edge_index = torch.stack([src_idx, dst_idx], dim=0)


def build_pairs_from_timelines(
    timelines: list,
    output_path: str | Path,
    k_days: int = 14,
    window_days: int = 7,
) -> Path:
    """
    Build and save a pairs file from a list of RepoTimeline objects.

    Args:
        timelines: list of core.dataset.graph_builder.RepoTimeline
        output_path: where to save the pickle
        k_days: prediction horizon in days (must be multiple of window_days)
        window_days: snapshot window size

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    k_steps = k_days // window_days  # how many windows ahead to predict

    pairs: list[tuple[Snapshot, Snapshot]] = []
    for timeline in timelines:
        snaps = timeline.get_snapshots(window_days=window_days, step_days=window_days)
        for i in range(len(snaps) - k_steps):
            pairs.append((snaps[i], snaps[i + k_steps]))

    log.info("Built %d pairs from %d timelines", len(pairs), len(timelines))
    with open(output_path, "wb") as f:
        pickle.dump(pairs, f)
    return output_path
