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


def configure_pyg_kernels(segment_matmul: str = "on") -> bool | None:
    """
    Pin PyG's HeteroLinear dispatch (used by HGTConv's relation-specific K/V).

    With ``use_segment_matmul = None`` PyG times pyg-lib ``segment_matmul``
    against a per-type Python loop on the first batch of each size magnitude and
    caches the winner per module. That one-shot timing is noisy under load; on
    the GH Archive graph it picked the loop in some runs: 9.3 s vs 0.76 s per HGT
    train step (batch 512, fan-out [10, 5], measured). "on" uses segment_matmul
    whenever pyg-lib provides it, "off" forces the loop, "auto" keeps PyG's timing.
    """
    import torch_geometric.backend as pyg_backend
    from torch_geometric.typing import WITH_SEGMM

    if segment_matmul == "auto":
        pyg_backend.use_segment_matmul = None
    elif segment_matmul == "on":
        pyg_backend.use_segment_matmul = True if WITH_SEGMM else False
    elif segment_matmul == "off":
        pyg_backend.use_segment_matmul = False
    else:
        raise ValueError("segment_matmul must be 'on', 'off' or 'auto'")
    return pyg_backend.use_segment_matmul


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
        import torch_geometric.backend as pyg_backend

        return {
            "backend": self.name,
            "segment_matmul": pyg_backend.use_segment_matmul,
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


_CUGRAPH_COMMS = False


def init_cugraph_single_process() -> None:
    """
    Process initialisation following the cuGraph-PyG 26.10 examples:
    torch.distributed (NCCL) -> RMM -> [cuGraph comms] -> WholeGraph.
    Works under ``torchrun`` or plain ``python`` (world_size = 1).

    cuGraph comms are only initialised when world_size > 1. With a single
    process, cugraph_pyg's GraphStore builds a ``pylibcugraph.SGGraph`` with a
    plain ``ResourceHandle`` (``GraphStore.is_multi_gpu`` is False), and the
    comms setup would require raft-dask / ucxx, which the ``cugraph-pyg-cu12``
    pip wheels do not install (found when first running this on a GPU).
    """
    global _CUGRAPH_INITIALISED, _CUGRAPH_COMMS
    if _CUGRAPH_INITIALISED:
        return
    os.environ.setdefault("RAPIDS_NO_INITIALIZE", "1")  # let PyTorch create the CUDA context
    import torch.distributed as dist

    if not dist.is_initialized():
        for k, v in {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29511", "RANK": "0",
                     "WORLD_SIZE": "1", "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1"}.items():
            os.environ.setdefault(k, v)
        dist.init_process_group("nccl")
        import atexit

        atexit.register(lambda: dist.is_initialized() and dist.destroy_process_group())
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    import cupy
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    rmm.reinitialize(devices=[local_rank], pool_allocator=False, managed_memory=True)
    cupy.cuda.Device(local_rank).use()
    cupy.cuda.set_allocator(rmm_cupy_allocator)

    if world > 1:
        from pylibcugraph.comms import cugraph_comms_create_unique_id, cugraph_comms_init

        uid = [cugraph_comms_create_unique_id()] if rank == 0 else [None]
        dist.broadcast_object_list(uid, src=0, device=torch.device("cuda", local_rank))
        cugraph_comms_init(rank, world, uid[0], local_rank)
        _CUGRAPH_COMMS = True

    from pylibwholegraph.torch.initialize import init as wm_init

    wm_init(rank, world, local_rank, torch.cuda.device_count())
    _CUGRAPH_INITIALISED = True


def shutdown_cugraph() -> None:
    global _CUGRAPH_INITIALISED, _CUGRAPH_COMMS
    if not _CUGRAPH_INITIALISED:
        return
    from pylibwholegraph.torch.initialize import finalize as wm_finalize

    wm_finalize()
    if _CUGRAPH_COMMS:
        from pylibcugraph.comms import cugraph_comms_shutdown

        cugraph_comms_shutdown()
        _CUGRAPH_COMMS = False
    _CUGRAPH_INITIALISED = False


def _tolerate_missing_cudf() -> None:
    """
    Workaround for cugraph-pyg 26.10 installed from pip without cudf.

    ``GraphStore._put_edge_index`` dispatches on
    ``isinstance(edge_index, (cupy.ndarray, cudf.Series))``; when cudf is not
    installed, ``cudf`` is a ``MissingModule`` whose attribute access raises, so
    even a plain torch tensor crashes the call. cudf is not a dependency of the
    ``cugraph-pyg-cu12`` wheel and is otherwise only used in type hints there.
    Replace it with a namespace whose ``Series`` no object is an instance of, so
    the isinstance checks are simply False for torch inputs.
    """
    import types

    try:
        import cugraph_pyg.data.graph_store as gs
        from cugraph_pyg.utils.imports import MissingModule
    except ImportError:
        return
    if isinstance(getattr(gs, "cudf", None), MissingModule):
        class _CudfNotInstalledSeries:  # no instances exist
            pass

        gs.cudf = types.SimpleNamespace(Series=_CudfNotInstalledSeries)


def hetero_to_cugraph_stores(data, time_attr: str | None = "time"):
    """
    Copy a PyG HeteroData into cuGraph-PyG stores.

    Each edge type is registered explicitly (both directions are already
    present as rev_* relations in our graphs). Edge timestamps go into the
    FeatureStore under (edge_type, time_attr); cuGraph-PyG requires them for
    *every* edge type when temporal sampling is enabled.
    """
    from cugraph_pyg.data import FeatureStore, GraphStore

    _tolerate_missing_cudf()
    # location="cuda": keep WholeGraph storage in device memory. The default
    # ("cpu") maps host memory through SysV shared-memory segments; on the
    # development laptop that path aborted ("Create host shared memory from IPC
    # key ... failed, Invalid argument") and leaked one segment per tensor on
    # every crash. The final cuGraph graph lives on the GPU in either case.
    graph_store = GraphStore(location="cuda")
    feature_store = FeatureStore(location="cuda")
    for et in data.edge_types:
        src, _, dst = et
        ei = data[et].edge_index
        if ei.numel() == 0:
            # cugraph-pyg 26.10 turns an empty edge index into a 0-byte WholeGraph
            # allocation, which aborts the process (cuMemAddressReserve ->
            # CUDA_ERROR_INVALID_VALUE on device, shmget EINVAL on host). An edge
            # type with no edges contributes nothing to sampling; skip it.
            log.info("cuGraph stores: skipping empty edge type %s", et)
            continue
        graph_store[et, "coo", False, (data[src].num_nodes, data[dst].num_nodes)] = ei
        if time_attr is not None:
            t = getattr(data[et], time_attr, None)
            if t is None:
                raise ValueError(f"edge type {et} has no '{time_attr}' attribute")
            feature_store[et, time_attr, None] = t
    for nt in data.node_types:
        feature_store[nt, "x", None] = data[nt].x
    return feature_store, graph_store


class _AlignedSeedLoader:
    """
    Wrap a cugraph_pyg heterogeneous NeighborLoader so that
    ``batch[seed_type].input_id[i]`` refers to seed row ``i`` (PyG convention).

    Measured with cugraph-pyg 26.10 on a GPU: with ``shuffle=True`` the seed rows
    ``n_id[:batch_size]`` come back sorted by node id while ``input_id`` keeps the
    shuffled order (the sampler source has "TODO figure out how to set input_id
    for heterogeneous output"). Using ``input_id`` as-is would pair logits with
    the wrong labels. The wrapper matches rows to inputs by node id (seeds are
    unique within a batch) and also sets ``seed_time``, which the heterogeneous
    output does not carry.
    """

    def __init__(self, loader, seed_type: str, seeds: torch.Tensor, seed_time: torch.Tensor | None):
        self.loader, self.seed_type = loader, seed_type
        self.invalid_edges = 0
        self._warned = False
        self.seeds = seeds.cpu()
        self.seed_time = None if seed_time is None else seed_time.cpu()

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            st = batch[self.seed_type]
            bs = int(st.batch_size)
            ids = st.input_id.cpu()
            nodes = self.seeds[ids]
            rows = st.n_id[:bs].cpu()
            sorted_nodes, perm = nodes.sort()
            pos = torch.searchsorted(sorted_nodes, rows).clamp_max(len(sorted_nodes) - 1)
            if (len(ids) != bs or torch.unique(nodes).numel() != bs
                    or not torch.equal(sorted_nodes[pos], rows)):
                raise RuntimeError("cuGraph batch: cannot align seed rows with input ids")
            aligned = ids[perm[pos]]
            st.input_id = aligned.to(st.n_id.device)
            if self.seed_time is not None:
                st.seed_time = self.seed_time[aligned].to(st.n_id.device)
            # Drop edges whose local indices fall outside the sampled node sets.
            # Measured with cugraph-pyg 26.10 on a GTX 1650: ~2 % of batches
            # (synthetic graph, fan-out [5, 3], batch 256) contain such edges;
            # fed to a GNN they trigger a device-side assert in index_select.
            # The count is exposed so callers can report it; correctness of the
            # remaining edges is NOT guaranteed (see docs/upstream_issue_cugraph_pyg.md).
            invalid = 0
            for et in batch.edge_types:
                store = batch[et]
                ei = store.edge_index
                if ei.numel() == 0:
                    continue
                n_src, n_dst = batch[et[0]].num_nodes, batch[et[2]].num_nodes
                ok = (ei[0] >= 0) & (ei[0] < n_src) & (ei[1] >= 0) & (ei[1] < n_dst)
                if not bool(ok.all()):
                    invalid += int((~ok).sum())
                    store.edge_index = ei[:, ok]
                    if "e_id" in store:
                        store.e_id = store.e_id[ok]
            batch.num_invalid_edges = invalid
            self.invalid_edges += invalid
            if invalid and not self._warned:
                log.warning("cuGraph batch contained %d out-of-range edges (dropped)", invalid)
                self._warned = True
            yield batch


class CuGraphBackend:
    """cugraph_pyg.loader.NeighborLoader over (FeatureStore, GraphStore). NOT VERIFIED here."""

    name = "cugraph"

    def __init__(self, data, device: str | torch.device = "cuda") -> None:
        ok, reason = cugraph_available()
        if not ok:
            raise RuntimeError(f"cuGraph backend unavailable: {reason}")
        init_cugraph_single_process()
        self.device = torch.device(device)
        self.edge_types = [et for et in data.edge_types if data[et].edge_index.numel() > 0]
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
        loader = NeighborLoader(
            self.stores,
            num_neighbors=spec.num_neighbors(self.edge_types),
            input_nodes=(seed_type, seeds),
            batch_size=spec.batch_size,
            shuffle=shuffle,
            # cugraph-pyg 26.10 estimates seeds per sampling call as
            # 0.11 * GPU memory / prod(per-hop fan-out summed over edge types);
            # with 30 edge types on a 4 GB GPU that is ~35 seeds, which floors to
            # 0 batches per call and crashes (torch.split with size 0). One batch
            # per call is always valid.
            local_seeds_per_call=spec.batch_size,
            **kwargs,
        )
        return _AlignedSeedLoader(loader, seed_type, seeds, seed_time if spec.temporal else None)

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
