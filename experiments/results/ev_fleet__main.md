# ev_fleet__main

- config: `configs/ev_fleet.toml`
- seeds: [0, 1, 2]
- data: ev_fleet_synthetic; seeds per split {'train': 7384, 'val': 1843, 'test': 1829}, positive rate {'train': 0.056, 'val': 0.066, 'test': 0.056}
- environment: {'backend': 'pyg', 'torch': '2.8.0+cpu', 'torch_geometric': '2.7.0', 'pyg_lib': '0.5.0+pt28cpu', 'cpu': '11th Gen Intel(R) Core(TM) i5-11300H @ 3.10GHz', 'cpu_threads': 4, 'git_commit': 'e05a40855e1066353646163ca53998254463dd56', 'git_dirty': True}

| Model | Test ROC-AUC | Test AP | Test F1 | Val ROC-AUC | Train s | Test seeds/s | Params |
|---|---|---|---|---|---|---|---|
| LogReg (intrinsic) | 0.788 ± 0.000 | 0.514 ± 0.000 | 0.578 ± 0.000 | 0.737 ± 0.000 | 0.0 | 5933761 | 15 |
| LightGBM (intrinsic) | 0.835 ± 0.014 | 0.639 ± 0.019 | 0.673 ± 0.033 | 0.810 ± 0.004 | 0.1 | 2261090 | 55 |
| MLP (intrinsic) | 0.791 ± 0.008 | 0.589 ± 0.007 | 0.620 ± 0.004 | 0.783 ± 0.004 | 2.2 | 47063 | 5441 |
| HeteroSAGE (intrinsic) | 0.869 ± 0.008 | 0.683 ± 0.008 | 0.670 ± 0.018 | 0.855 ± 0.003 | 54.8 | 4750 | 334209 |
| HGT (intrinsic) | 0.863 ± 0.016 | 0.644 ± 0.030 | 0.652 ± 0.037 | 0.832 ± 0.005 | 68.5 | 2780 | 400863 |
| LogReg (+history) | 0.899 ± 0.000 | 0.562 ± 0.000 | 0.554 ± 0.000 | 0.842 ± 0.000 | 0.0 | 5556583 | 20 |
| LightGBM (+history) | 0.931 ± 0.003 | 0.746 ± 0.003 | 0.675 ± 0.004 | 0.883 ± 0.004 | 0.2 | 905247 | 46 |
| MLP (+history) | 0.917 ± 0.004 | 0.684 ± 0.018 | 0.603 ± 0.018 | 0.882 ± 0.004 | 2.7 | 43578 | 5761 |
| HeteroSAGE (+history) | 0.926 ± 0.003 | 0.715 ± 0.018 | 0.626 ± 0.022 | 0.898 ± 0.005 | 27.6 | 4912 | 334529 |
| HGT (+history) | 0.919 ± 0.005 | 0.725 ± 0.012 | 0.608 ± 0.018 | 0.894 ± 0.002 | 81.5 | 2825 | 401183 |

Mean ± std over seeds (model initialisation, batch order, neighbour sampling); the graph, the seeds and the splits are fixed. Thresholds for F1 are chosen on validation. Params: trainable parameters (LightGBM: number of trees).

## Paired comparisons (same test seeds)

| A | B | AUC(A) − AUC(B) | 95% paired bootstrap CI | P(diff ≤ 0) |
|---|---|---|---|---|
| HeteroSAGE (intrinsic) | MLP (intrinsic) | +0.0783 | [+0.0489, +0.1125] | 0.000 |
| HGT (intrinsic) | MLP (intrinsic) | +0.0722 | [+0.0396, +0.1091] | 0.000 |
| HGT (intrinsic) | HeteroSAGE (intrinsic) | -0.0061 | [-0.0287, +0.0165] | 0.727 |
| LightGBM (+history) | LightGBM (intrinsic) | +0.0958 | [+0.0605, +0.1351] | 0.000 |
| HeteroSAGE (+history) | LightGBM (+history) | -0.0044 | [-0.0206, +0.0116] | 0.704 |
| HGT (+history) | LightGBM (+history) | -0.0118 | [-0.0291, +0.0057] | 0.903 |
| HGT (+history) | MLP (+history) | +0.0022 | [-0.0067, +0.0121] | 0.310 |
| HGT (+history) | HeteroSAGE (+history) | -0.0074 | [-0.0191, +0.0041] | 0.895 |

Mean over model seeds of the per-seed AUC difference; 1,000 bootstrap resamples of the test seeds shared by both models.
