# Draft upstream issue: `HGTConv` output depends on the iteration order of `edge_index_dict`

Status: **not filed**. Draft for `pyg-team/pytorch_geometric`, verified on
torch_geometric 2.7.0 (torch 2.8.0+cpu, pyg-lib 0.5.0); the same code is on
`master` as of 2026-10-12.

## 🐛 Describe the bug

`HGTConv._construct_src_node_feat` concatenates source features and builds the
relation `type_vec` in the iteration order of `edge_index_dict`, then calls
`self.k_rel` / `self.v_rel`, which are `HeteroLinear(..., is_sorted=True)`.
`type_vec` is only sorted if `edge_index_dict` iterates in the same order as
`metadata[1]`. Otherwise:

* naive path (`torch_geometric.backend.use_segment_matmul = False`, or the
  timing cache picks it): `index2ptr` on an unsorted `type_vec` yields wrong
  segment boundaries, so relation-specific K/V weights are applied to the
  wrong rows — **silently**;
* segment-matmul path: `pyg_lib.ops.segment_matmul` raises
  `split_with_sizes expects split_sizes have only non-negative entries`.

The order mismatch is common in practice: graphs built by inserting forward and
reverse relations pairwise (`(a, r, b)`, `(b, rev_r, a)`, ...) have a
`data.edge_types` order that differs from a metadata list declared as "all
forward, then all reverse", and `NeighborLoader` batches inherit the data
order.

```python
import torch
import torch_geometric.backend as backend
from torch_geometric.nn import HGTConv

torch.manual_seed(0)
node_types = ["a", "b"]
edge_types = [("a", "r1", "b"), ("b", "r2", "a"), ("a", "r3", "a")]
conv = HGTConv(8, 8, (node_types, edge_types), heads=2)
x = {"a": torch.randn(5, 8), "b": torch.randn(4, 8)}
ei = {("a", "r1", "b"): torch.tensor([[0, 1, 2], [0, 1, 3]]),
      ("b", "r2", "a"): torch.tensor([[0, 3], [4, 2]]),
      ("a", "r3", "a"): torch.tensor([[1, 2], [0, 3]])}
backend.use_segment_matmul = False
out1 = conv(x, ei)
out2 = conv(x, {k: ei[k] for k in [edge_types[2], edge_types[0], edge_types[1]]})
print((out1["a"] - out2["a"]).abs().max(), (out1["b"] - out2["b"]).abs().max())
# tensor(0.1413) tensor(0.2797)   -> expected 0
```

## Suggested fix

Iterate `self.edge_types` (filtered to the keys present) instead of
`edge_index_dict.keys()` in `_construct_src_node_feat` (and use the same order
when splitting the result), or construct `k_rel`/`v_rel` with
`is_sorted=False`.

## Workaround

Pass `{et: edge_index_dict[et] for et in metadata[1] if et in edge_index_dict}`
(see `models/hetero_gnn.py::ordered_edges` in this repository).
