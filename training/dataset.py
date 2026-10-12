"""
PyTorch Dataset for the graph-level JEPA training pairs.

Each sample is a (T, T+k) snapshot pair from the same repo.
The DataLoader yields batches of:
    graph_t     HeteroData   — project state during [T, T+w)
    graph_t_k   HeteroData   — project state during [T+k, T+k+w) (the target)
    events      Tensor       — (EVENT_FEAT_DIM,) summary of events in the gap
                               [T+w, T+k); zeros when the windows are adjacent
    labels      Tensor       — (3,) [regression, velocity_drop, release_in_t_k]

The dataset is built offline from GH Archive data using core/dataset/.
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

try:
    from torch_geometric.data import Batch, HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from core.dataset.graph_builder import (
    EVENT_FEAT_DIM,
    NODE_FEAT_DIMS,
    NODE_TYPES,
    Snapshot,
)
from core.dataset.labeler import label_pair

log = logging.getLogger(__name__)


class DeliveryPairDataset(Dataset):
    """
    Dataset of (snapshot_T, snapshot_T+k) pairs with labels.

    Args:
        pairs_path: pickle produced by build_pairs_from_timelines (list of dicts
                    with keys "snap_t", "snap_t_k", "gap_events"). Legacy files
                    containing (Snapshot, Snapshot) tuples are accepted; their
                    gap-event vector is set to zeros.
    """

    def __init__(self, pairs_path: str | Path, k_days: int = 14) -> None:
        self.k_days = k_days
        pairs_path = Path(pairs_path)
        log.info("Loading pairs from %s", pairs_path)
        with open(pairs_path, "rb") as f:
            raw = pickle.load(f)

        self.samples: list[tuple[Snapshot, Snapshot, np.ndarray]] = []
        legacy = 0
        for item in raw:
            if isinstance(item, dict):
                gap = item.get("gap_events")
                if gap is None:
                    gap = np.zeros(EVENT_FEAT_DIM, dtype=np.float32)
                self.samples.append((item["snap_t"], item["snap_t_k"], gap))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                legacy += 1
                self.samples.append((item[0], item[1], np.zeros(EVENT_FEAT_DIM, dtype=np.float32)))
        if legacy:
            log.warning("%d legacy pairs without gap events: predictor conditioning set to zeros", legacy)

        log.info("Loaded %d training pairs", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        snap_t, snap_t_k, gap = self.samples[idx]
        lb = label_pair(snap_t, snap_t_k)
        return {
            "graph_t": snapshot_to_hetero(snap_t),
            "graph_t_k": snapshot_to_hetero(snap_t_k),
            "events": torch.as_tensor(gap, dtype=torch.float32),
            "labels": torch.tensor(
                [lb["regression"], lb["velocity_drop"], lb["release_in_t_k"]],
                dtype=torch.float32,
            ),
            "repo": snap_t.repo,
        }

    def temporal_split(self, val_fraction: float = 0.1) -> tuple[Subset, Subset]:
        """
        Split by time so that no validation information is visible in training.

        A cutoff C is chosen so that ~val_fraction of the pairs have their
        context window starting at or after C. Training keeps only pairs whose
        *target* window ends at or before C, so the training targets never
        overlap the validation contexts. Pairs straddling C are dropped.
        """
        starts = sorted(s[0].t_start for s in self.samples)
        if len(starts) < 2:
            raise ValueError("need at least 2 pairs for a temporal split")
        cutoff = starts[min(len(starts) - 1, int(len(starts) * (1 - val_fraction)))]
        train_idx = [i for i, s in enumerate(self.samples) if s[1].t_end <= cutoff]
        val_idx = [i for i, s in enumerate(self.samples) if s[0].t_start >= cutoff]
        log.info(
            "Temporal split at %s: train=%d val=%d dropped=%d",
            cutoff, len(train_idx), len(val_idx), len(self.samples) - len(train_idx) - len(val_idx),
        )
        return Subset(self, train_idx), Subset(self, val_idx)


def snapshot_to_hetero(snap: Snapshot) -> "HeteroData":
    """Convert a Snapshot into a PyG HeteroData object with observed edges only."""
    if not HAS_PYG:
        raise ImportError("torch-geometric required")

    data = HeteroData()
    feats = snap.to_node_features()
    for ntype in NODE_TYPES:
        x = feats.get(ntype)
        if x is None:
            x = np.zeros((1, NODE_FEAT_DIMS[ntype]), dtype=np.float32)
        data[ntype].x = torch.as_tensor(x, dtype=torch.float32)
    for etype, ei in snap.to_edge_index().items():
        data[etype].edge_index = torch.as_tensor(ei, dtype=torch.long)
    return data


# Backward-compatible alias (the previous name was private).
_snapshot_to_hetero = snapshot_to_hetero


def collate_fn(batch: list[dict]) -> dict:
    """PyG batches the graphs; events and labels are stacked tensors."""
    if not HAS_PYG:
        raise ImportError("torch-geometric required for training")
    return {
        "graph_t": Batch.from_data_list([item["graph_t"] for item in batch]),
        "graph_t_k": Batch.from_data_list([item["graph_t_k"] for item in batch]),
        "events": torch.stack([item["events"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def build_pairs_from_timelines(
    timelines: list,
    output_path: str | Path,
    k_days: int = 14,
    window_days: int = 7,
    min_events: int = 5,
) -> Path:
    """
    Build and save a pairs file from a list of RepoTimeline objects.

    Pairs are matched by exact time offset (t, t + k_days), not by list index:
    windows with fewer than ``min_events`` events are skipped by
    RepoTimeline.get_snapshots, so index arithmetic would silently pair
    windows that are further apart than k_days.

    Args:
        timelines: list of core.dataset.graph_builder.RepoTimeline
        output_path: where to save the pickle
        k_days: offset between the starts of context and target windows
        window_days: snapshot window size (k_days must be >= window_days)

    Returns:
        Path to the saved file.
    """
    if k_days < window_days:
        raise ValueError("k_days must be >= window_days (context and target windows must not overlap)")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs: list[dict] = []
    for timeline in timelines:
        snaps = timeline.get_snapshots(
            window_days=window_days, step_days=window_days, min_events=min_events
        )
        by_start: dict[datetime, Snapshot] = {s.t_start: s for s in snaps}
        for snap in snaps:
            target = by_start.get(snap.t_start + timedelta(days=k_days))
            if target is None:
                continue
            gap_events = [
                ev for ev in timeline.events
                if snap.t_end <= _ts(ev["timestamp"]) < target.t_start
            ]
            gap_vec = Snapshot(timeline.repo, snap.t_end, target.t_start, gap_events).to_event_vector()
            pairs.append({"snap_t": snap, "snap_t_k": target, "gap_events": gap_vec})

    log.info("Built %d pairs from %d timelines", len(pairs), len(timelines))
    with open(output_path, "wb") as f:
        pickle.dump(pairs, f)
    return output_path


def _ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
