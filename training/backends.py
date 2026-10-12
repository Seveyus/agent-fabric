"""
Graph sampling backends: native PyG (CPU or GPU) and NVIDIA cuGraph-PyG (GPU).

Both backends return mini-batches as ``torch_geometric.data.HeteroData`` with
the same seed bookkeeping, so the model and training loop are backend-agnostic:

    batch[seed_type].batch_size   number of seeds in the batch
    batch[seed_type].input_id     position of each seed in the seed array
    batch[seed_type].n_id         global node ids of sampled nodes
    first `batch_size` rows of batch[seed_type].x are the seeds, in input order

What runs where
---------------
``pyg``      Graph topology (CSC) and features live in host memory. Neighbour
             sampling runs in pyg-lib (C++, CPU threads). Feature gather runs in
             PyTorch on CPU; the batch is then copied to ``device``.
``cugraph``  The graph is built as a cuGraph graph in GPU memory; features (and
             edge timestamps) are held in a WholeGraph-backed FeatureStore.
             Neighbour sampling runs on the GPU (pylibcugraph); the sampled
             subgraph and gathered features are produced on the GPU. Model
             forward/backward is plain PyTorch in both cases.

Temporal semantics (both backends never sample an edge newer than the seed):
``pyg``      every sampled edge satisfies ``edge_time <= seed_time`` (all hops
             compare to the seed's time). Requires pyg-lib; disjoint subgraphs.
``cugraph``  ``temporal_comparison='monotonically_decreasing'`` (cuGraph-PyG
             default): edge times must not increase along a sampled path,
             starting from the seed time. This is a strict subset of the PyG
             candidate set, so the two backends do not sample identical
             neighbourhoods; compare model quality, not batches.

STATUS of the cuGraph backend: implemented against the cuGraph-PyG 26.10 API
(GraphStore / FeatureStore / loader.NeighborLoader, see docs/GRAPH_LEARNING.md)
but NOT EXECUTED in this repository's development environment (no usable
NVIDIA driver/CUDA runtime). Treat it as unverified until
``pytest -m cugraph`` and ``benchmarks/sampling_throughput.py --backend cugraph``
have been run on a RAPIDS-capable machine.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Protocol

import torch

log = logging.getLogger(__name__)

EdgeType = tuple[str, str, str]


@dataclass(frozen=True)
class SamplerSpec:
    """Fan-out per edge type and hop; edge types not listed get ``default_fanout``."""

    fanout: dict[EdgeType, list[int]] | None
    default_fanout: list[int]
    batch_size: int = 512
    temporal: bool = True
    time_attr: str = "time"

    @property
    def num_hops(self) -> int:
        return len(self.default_fanout)

    def num_neighbors(self, edge_types: list[EdgeType]) -> dict[EdgeType, list[int]]:
        out = {}
        for et in edge_types:
            f = (self.fanout or {}).get(et, self.default_fanout)
            if len(f) != self.num_hops:
                raise ValueError(f"fanout for {et} has {len(f)} hops, expected {self.num_hops}")
            out[et] = list(f)
        return out


class GraphBackend(Protocol):
    name: str
    device: torch.device

    def loader(
        self,
        seed_type: str,
        seeds: torch.Tensor,
        seed_time: torch.Tensor | None,
        spec: SamplerSpec,
        shuffle: bool,
        generator: torch.Generator | None = None,
    ) -> Iterable: ...

    def describe(self) -> dict: ...


# --------------------------------------------------------------------------- #
# Native PyG                                                                  #
# --------------------------------------------------------------------------- #

class PyGBackend:
    """torch_geometric.loader.NeighborLoader over an in-memory HeteroData."""

    name = "pyg"

    def __init__(self, data, device: str | torch.device = "cpu", num_workers: int = 0) -> None:
        from torch_geometric.typing import WITH_EDGE_TIME_NEIGHBOR_SAMPLE

        self.data = data
        self.device = torch.device(device)
        self.num_workers = num_workers
        self._edge_time_ok = bool(WITH_EDGE_TIME_NEIGHBOR_SAMPLE)

    def loader(self, seed_type, seeds, seed_time, spec, shuffle, generator=None):
        from torch_geometric.loader import NeighborLoader

        kwargs = {}
        if spec.temporal:
            if seed_time is None:
                raise ValueError("temporal sampling requires seed times")
            if not self._edge_time_ok:
                raise RuntimeError(
                    "edge-level temporal sampling needs pyg-lib (>= 0.4) — "
                    "install it from https://data.pyg.org/whl/ for your torch version"
                )
            kwargs.update(input_time=seed_time, time_attr=spec.time_attr, temporal_strategy="uniform")
        return NeighborLoader(
            self.data,
            num_neighbors=spec.num_neighbors(self.data.edge_types),
            input_nodes=(seed_type, seeds),
            batch_size=spec.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            generator=generator,
            **kwargs,
        )

    def describe(self) -> dict:
        import torch_geometric

        try:
            import pyg_lib
            pyg_lib_version = pyg_lib.__version__
        except ImportError:
            pyg_lib_version = None
        return {
            "backend": self.name,
            "sampling": "pyg-lib on CPU" if pyg_lib_version else "torch-sparse on CPU",
            "feature_gather": "PyTorch, host memory",
            "model_device": str(self.device),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "pyg_lib": pyg_lib_version,
            "cpu_threads": torch.get_num_threads(),
        }


# --------------------------------------------------------------------------- #
# NVIDIA cuGraph-PyG                                                          #
# --------------------------------------------------------------------------- #

def cugraph_available() -> tuple[bool, str]:
    """(ok, reason). Checks imports and a usable CUDA device without side effects."""
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False (no CUDA build of torch or no driver)"
    try:
        import cugraph_pyg  # noqa: F401
        import pylibcugraph  # noqa: F401
        import pylibwholegraph  # noqa: F401
    except ImportError as exc:
        return False, f"cuGraph-PyG stack not importable: {exc}"
    major, _ = torch.cuda.get_device_capability()
    if major < 7:
        return False, "RAPIDS requires compute capability >= 7.0 (Volta or newer)"
    return True, "ok"


_CUGRAPH_INITIALISED = False


def init_cugraph_single_process() -> None:
    """
    Single-process (world_size=1) initialisation following the cuGraph-PyG
    26.10 examples: torch.distributed (NCCL) -> cuGraph comms -> RMM -> WholeGraph.
    Works under ``torchrun --nproc-per-node=1`` or plain ``python``.
    """
    global _CUGRAPH_INITIALISED
    if _CUGRAPH_INITIALISED:
        return
    os.environ.setdefault("RAPIDS_NO_INITIALIZE", "1")  # let PyTorch create the CUDA context
    import torch.distributed as dist

    if not dist.is_initialized():
        for k, v in {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29511", "RANK": "0",
                     "WORLD_SIZE": "1", "LOCAL_RANK": "0"}.items():
            os.environ.setdefault(k, v)
        dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    from pylibcugraph.comms import cugraph_comms_create_unique_id, cugraph_comms_init

    uid = [cugraph_comms_create_unique_id()] if rank == 0 else [None]
    dist.broadcast_object_list(uid, src=0, device=torch.device("cuda", local_rank))

    import cupy
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    rmm.reinitialize(devices=[local_rank], pool_allocator=False, managed_memory=True)
    cupy.cuda.Device(local_rank).use()
    cupy.cuda.set_allocator(rmm_cupy_allocator)

    cugraph_comms_init(rank, world, uid[0], local_rank)

    from pylibwholegraph.torch.initialize import init as wm_init

    wm_init(rank, world, local_rank, torch.cuda.device_count())
    _CUGRAPH_INITIALISED = True


def shutdown_cugraph() -> None:
    global _CUGRAPH_INITIALISED
    if not _CUGRAPH_INITIALISED:
        return
    from pylibcugraph.comms import cugraph_comms_shutdown
    from pylibwholegraph.torch.initialize import finalize as wm_finalize

    wm_finalize()
    cugraph_comms_shutdown()
    _CUGRAPH_INITIALISED = False


def hetero_to_cugraph_stores(data, time_attr: str | None = "time"):
    """
    Copy a PyG HeteroData into cuGraph-PyG stores.

    Each edge type is registered explicitly (both directions are already
    present as rev_* relations in our graphs). Edge timestamps go into the
    FeatureStore under (edge_type, time_attr); cuGraph-PyG requires them for
    *every* edge type when temporal sampling is enabled.
    """
    from cugraph_pyg.data import FeatureStore, GraphStore

    graph_store = GraphStore()
    feature_store = FeatureStore()
    for et in data.edge_types:
        src, _, dst = et
        ei = data[et].edge_index
        graph_store[et, "coo", False, (data[src].num_nodes, data[dst].num_nodes)] = ei
        if time_attr is not None:
            t = getattr(data[et], time_attr, None)
            if t is None:
                raise ValueError(f"edge type {et} has no '{time_attr}' attribute")
            feature_store[et, time_attr, None] = t
    for nt in data.node_types:
        feature_store[nt, "x", None] = data[nt].x
    return feature_store, graph_store


class CuGraphBackend:
    """cugraph_pyg.loader.NeighborLoader over (FeatureStore, GraphStore). NOT VERIFIED here."""

    name = "cugraph"

    def __init__(self, data, device: str | torch.device = "cuda") -> None:
        ok, reason = cugraph_available()
        if not ok:
            raise RuntimeError(f"cuGraph backend unavailable: {reason}")
        init_cugraph_single_process()
        self.device = torch.device(device)
        self.edge_types = list(data.edge_types)
        self.stores = hetero_to_cugraph_stores(data)

    def loader(self, seed_type, seeds, seed_time, spec, shuffle, generator=None):
        from cugraph_pyg.loader import NeighborLoader

        kwargs = {}
        if spec.temporal:
            if seed_time is None:
                raise ValueError("temporal sampling requires seed times")
            kwargs.update(
                input_time=seed_time,
                time_attr=spec.time_attr,
                temporal_strategy="uniform",
                temporal_comparison="monotonically_decreasing",
                disjoint=True,  # same as PyG temporal sampling
            )
        if generator is not None:
            log.debug("cuGraph loader ignores the torch.Generator; shuffling uses its own RNG")
        return NeighborLoader(
            self.stores,
            num_neighbors=spec.num_neighbors(self.edge_types),
            input_nodes=(seed_type, seeds),
            batch_size=spec.batch_size,
            shuffle=shuffle,
            **kwargs,
        )

    def describe(self) -> dict:
        import cugraph_pyg

        return {
            "backend": self.name,
            "sampling": "pylibcugraph on GPU (temporal: monotonically_decreasing)",
            "feature_gather": "WholeGraph FeatureStore",
            "model_device": str(self.device),
            "torch": torch.__version__,
            "cugraph_pyg": getattr(cugraph_pyg, "__version__", "unknown"),
            "gpu": torch.cuda.get_device_name(),
        }


def make_backend(name: str, data, device: str | None = None, **kwargs) -> GraphBackend:
    if name == "pyg":
        return PyGBackend(data, device=device or "cpu", **kwargs)
    if name == "cugraph":
        return CuGraphBackend(data, device=device or "cuda", **kwargs)
    raise ValueError(f"unknown backend {name!r} (expected 'pyg' or 'cugraph')")
