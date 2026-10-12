# synthetic_pr_merge__main

- config: `configs/synthetic_pr_merge.toml`
- seeds: [0, 1, 2]
- data: synthetic; seeds per split {'train': 2982, 'val': 957, 'test': 1461}, positive rate {'train': 0.458, 'val': 0.508, 'test': 0.472}
- environment: {'backend': 'pyg', 'torch': '2.8.0+cpu', 'torch_geometric': '2.7.0', 'pyg_lib': '0.5.0+pt28cpu', 'cpu': '11th Gen Intel(R) Core(TM) i5-11300H @ 3.10GHz', 'cpu_threads': 4, 'git_commit': 'aa16ff079bce03508899b87f0aba2cbc296b84e0', 'git_dirty': True}

| Model | Test ROC-AUC | Test AP | Test F1 | Val ROC-AUC | Train s | Test seeds/s | Params |
|---|---|---|---|---|---|---|---|
| LogReg (intrinsic) | 0.533 ± 0.000 | 0.502 ± 0.000 | 0.642 ± 0.000 | 0.526 ± 0.000 | 0.0 | 4699931 | 24 |
| LightGBM (intrinsic) | 0.514 ± 0.010 | 0.484 ± 0.009 | 0.642 ± 0.000 | 0.529 ± 0.002 | 0.1 | 3625761 | 14 |
| MLP (intrinsic) | 0.526 ± 0.005 | 0.501 ± 0.008 | 0.638 ± 0.002 | 0.516 ± 0.002 | 0.7 | 39201 | 6017 |
| HeteroSAGE (intrinsic) | 0.692 ± 0.004 | 0.642 ± 0.005 | 0.671 ± 0.001 | 0.680 ± 0.006 | 15.7 | 3788 | 531073 |
| HGT (intrinsic) | 0.702 ± 0.008 | 0.649 ± 0.011 | 0.676 ± 0.005 | 0.701 ± 0.009 | 22.4 | 2104 | 364807 |
| LogReg (+history) | 0.739 ± 0.000 | 0.704 ± 0.000 | 0.694 ± 0.000 | 0.746 ± 0.000 | 0.0 | 4334402 | 34 |
| LightGBM (+history) | 0.714 ± 0.004 | 0.671 ± 0.006 | 0.686 ± 0.001 | 0.730 ± 0.004 | 0.1 | 2647567 | 67 |
| MLP (+history) | 0.728 ± 0.005 | 0.685 ± 0.010 | 0.686 ± 0.007 | 0.731 ± 0.002 | 0.9 | 38164 | 6657 |
| HeteroSAGE (+history) | 0.730 ± 0.002 | 0.691 ± 0.004 | 0.690 ± 0.007 | 0.735 ± 0.001 | 7.5 | 3835 | 531713 |
| HGT (+history) | 0.727 ± 0.004 | 0.691 ± 0.007 | 0.687 ± 0.007 | 0.740 ± 0.002 | 18.0 | 2177 | 365447 |

Mean ± std over seeds (model initialisation, batch order, neighbour sampling); the graph, the seeds and the splits are fixed. Thresholds for F1 are chosen on validation. Params: trainable parameters (LightGBM: number of trees).

## Paired comparisons (same test seeds)

| A | B | AUC(A) − AUC(B) | 95% paired bootstrap CI | P(diff ≤ 0) |
|---|---|---|---|---|
| HeteroSAGE (intrinsic) | MLP (intrinsic) | +0.1664 | [+0.1352, +0.1965] | 0.000 |
| HGT (intrinsic) | MLP (intrinsic) | +0.1761 | [+0.1446, +0.2071] | 0.000 |
| HGT (intrinsic) | HeteroSAGE (intrinsic) | +0.0097 | [-0.0028, +0.0224] | 0.047 |
| LightGBM (+history) | LightGBM (intrinsic) | +0.1998 | [+0.1623, +0.2374] | 0.000 |
| HeteroSAGE (+history) | LightGBM (+history) | +0.0161 | [+0.0057, +0.0266] | 0.002 |
| HGT (+history) | LightGBM (+history) | +0.0132 | [+0.0028, +0.0235] | 0.008 |
| HGT (+history) | MLP (+history) | -0.0009 | [-0.0061, +0.0044] | 0.590 |
| HGT (+history) | HeteroSAGE (+history) | -0.0030 | [-0.0079, +0.0020] | 0.855 |

Mean over model seeds of the per-seed AUC difference; 1,000 bootstrap resamples of the test seeds shared by both models.
