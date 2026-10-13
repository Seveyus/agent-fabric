# Draft upstream issues: cugraph-pyg 26.10 heterogeneous temporal sampling

Status: **not filed**. Draft for `rapidsai/cugraph-gnn`. Observed with
cugraph-pyg-cu12 26.10.0, pylibcugraph-cu12 26.10.0, pylibwholegraph-cu12
26.10.0, torch 2.8.0+cu129, torch-geometric 2.7.0, single process (world size 1),
pip install (no conda), GeForce GTX 1650 Max-Q (CC 7.5, 4 GB), driver 595.99.02,
Ubuntu 24.04, kernel 7.0.0-38. Reproduction: `scripts/cugraph_temporal_check.py`
in this repository (synthetic heterogeneous graph: 3 node types, 30 non-empty
edge types with int64 edge times; one seed per batch so that every sampled edge
belongs to that seed). Raw numbers: `experiments/results/cugraph_temporal_check.json`.

## 1. Two-hop temporal sampling returns edges newer than the seed time

`NeighborLoader(..., input_time=seed_time, time_attr="time",
temporal_strategy="uniform", temporal_comparison="monotonically_decreasing")`

| fan-out | disjoint | sampled edges (60 seeds) | edges with time > seed time |
|---|---|---|---|
| [6] | True / False | 120 / 120 | 0 / 0 |
| [6, 3] | True | 1,850 | 16 (0.9 %) |
| [6, 3] | False | 1,845 | 12 (0.7 %) |

Edge times are looked up in the original graph through the returned `e_id`
(and, independently, as the earliest edge between the reported endpoints).
One-hop sampling is always correct; violations appear at hop 2. With a
path-monotone rule (each hop no newer than the previous one, starting from the
seed time) a hop-2 edge newer than the seed should be impossible.

## 2. `e_id` sometimes disagrees with the reported endpoints

For 12–16 of ~1,850 two-hop edges, `data[edge_type].edge_index[:, e_id]` is
not the pair `(n_id_src[edge_index[0]], n_id_dst[edge_index[1]])` returned in
the batch; in the examples the source differs by one node id within the same
destination neighbourhood (e.g. reported PR 47 → repo 1, `e_id` points to
PR 48 → repo 1). Never observed at hop 1.

## 3. `temporal_comparison="strictly_decreasing"` raises `KeyError: 'renumber_map'`

For every fan-out and both `disjoint` values, on the first batch.

## 4. Heterogeneous output: `input_id` is not aligned with the seed rows when `shuffle=True`

The seed rows `batch[seed_type].n_id[:batch_size]` come back sorted by node id
while `batch[seed_type].input_id` keeps the shuffled input order (the sampler
source says "TODO figure out how to set input_id for heterogeneous output").
Code that follows the PyG convention (`labels[input_id]` paired with the first
`batch_size` rows) silently trains on mismatched labels. `seed_time` is also
not set on heterogeneous batches. Workaround here:
`training/backends.py::_AlignedSeedLoader` (matches rows to inputs by node id).

## 5. Installation / initialisation gaps found on the pip path

* `pylibcugraph.comms.cugraph_comms_create_unique_id()` needs raft-dask/ucxx,
  which the `cugraph-pyg-cu12` wheel does not pull in. For world size 1 the
  comms are not needed (`GraphStore` builds an `SGGraph`), but the examples call
  them unconditionally.
* `GraphStore._put_edge_index` evaluates `isinstance(x, (cupy.ndarray, cudf.Series))`;
  without cudf installed, `cudf` is a `MissingModule` and the attribute access
  raises even for torch tensors.
* An edge type with zero edges creates a 0-byte WholeGraph tensor and aborts
  the process (`cuMemAddressReserve` → `CUDA_ERROR_INVALID_VALUE` with
  `location="cuda"`; `shmget` → `EINVAL` with the default `location="cpu"`).
  With the default host location, every abort leaks the SysV shared-memory
  segments already created (360 left behind after three crashed runs here).
