# gh_pr_merge__main

- config: `configs/gh_pr_merge.toml`
- seeds: [0, 1, 2]
- data: gharchive; seeds per split {'train': 40000, 'val': 15000, 'test': 15000}, positive rate {'train': 0.324, 'val': 0.311, 'test': 0.32}
- environment: {'backend': 'pyg', 'torch': '2.8.0+cpu', 'torch_geometric': '2.7.0', 'pyg_lib': '0.5.0+pt28cpu', 'cpu': '11th Gen Intel(R) Core(TM) i5-11300H @ 3.10GHz', 'cpu_threads': 4, 'git_commit': 'b0653b7807f6ce03ca4e77bfd3330e8ff9654d75', 'git_dirty': True}

| Model | Test ROC-AUC | Test AP | Test F1 | Val ROC-AUC | Train s | Test seeds/s | Params |
|---|---|---|---|---|---|---|---|
| LogReg (intrinsic) | 0.648 ± 0.000 | 0.433 ± 0.000 | 0.521 ± 0.000 | 0.614 ± 0.000 | 0.1 | 10511175 | 24 |
| LightGBM (intrinsic) | 0.681 ± 0.000 | 0.460 ± 0.001 | 0.543 ± 0.001 | 0.634 ± 0.001 | 0.3 | 1486605 | 35 |
| MLP (intrinsic) | 0.678 ± 0.001 | 0.457 ± 0.004 | 0.538 ± 0.002 | 0.631 ± 0.001 | 6.4 | 86046 | 6017 |
| HeteroSAGE (intrinsic) | 0.721 ± 0.002 | 0.508 ± 0.001 | 0.564 ± 0.004 | 0.675 ± 0.002 | 94.9 | 5703 | 531073 |
| HGT (intrinsic) | 0.715 ± 0.002 | 0.499 ± 0.007 | 0.560 ± 0.001 | 0.673 ± 0.001 | 165.7 | 3144 | 364807 |
| LogReg (+history) | 0.718 ± 0.000 | 0.514 ± 0.000 | 0.565 ± 0.000 | 0.673 ± 0.000 | 0.7 | 3770414 | 34 |
| LightGBM (+history) | 0.722 ± 0.001 | 0.520 ± 0.001 | 0.563 ± 0.001 | 0.675 ± 0.001 | 0.4 | 1072678 | 58 |
| MLP (+history) | 0.724 ± 0.001 | 0.512 ± 0.001 | 0.565 ± 0.001 | 0.677 ± 0.002 | 5.1 | 83263 | 6657 |
| HeteroSAGE (+history) | 0.730 ± 0.001 | 0.520 ± 0.003 | 0.570 ± 0.002 | 0.683 ± 0.002 | 87.3 | 5604 | 531713 |
| HGT (+history) | 0.728 ± 0.001 | 0.523 ± 0.001 | 0.567 ± 0.000 | 0.683 ± 0.002 | 165.9 | 3022 | 365447 |

Mean ± std over seeds (model initialisation, batch order, neighbour sampling); the graph, the seeds and the splits are fixed. Thresholds for F1 are chosen on validation. Params: trainable parameters (LightGBM: number of trees).

## Paired comparisons (same test seeds)

| A | B | AUC(A) − AUC(B) | 95% paired bootstrap CI | P(diff ≤ 0) |
|---|---|---|---|---|
| HeteroSAGE (intrinsic) | MLP (intrinsic) | +0.0431 | [+0.0368, +0.0488] | 0.000 |
| HGT (intrinsic) | MLP (intrinsic) | +0.0377 | [+0.0321, +0.0432] | 0.000 |
| HGT (intrinsic) | HeteroSAGE (intrinsic) | -0.0054 | [-0.0082, -0.0024] | 1.000 |
| LightGBM (+history) | LightGBM (intrinsic) | +0.0417 | [+0.0352, +0.0481] | 0.000 |
| HeteroSAGE (+history) | LightGBM (+history) | +0.0075 | [+0.0038, +0.0111] | 0.000 |
| HGT (+history) | LightGBM (+history) | +0.0052 | [+0.0016, +0.0091] | 0.002 |
| HGT (+history) | MLP (+history) | +0.0032 | [+0.0006, +0.0060] | 0.011 |
| HGT (+history) | HeteroSAGE (+history) | -0.0023 | [-0.0043, -0.0003] | 0.988 |

Mean over model seeds of the per-seed AUC difference; 1,000 bootstrap resamples of the test seeds shared by both models.
