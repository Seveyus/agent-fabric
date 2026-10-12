# synthetic_pr_merge__jepa

- config: `configs/synthetic_pr_merge.toml`
- seeds: [0, 1, 2]
- data: synthetic; seeds per split {'train': 2982, 'val': 957, 'test': 1461}, positive rate {'train': 0.458, 'val': 0.508, 'test': 0.472}
- environment: {'backend': 'pyg', 'torch': '2.8.0+cpu', 'torch_geometric': '2.7.0', 'pyg_lib': '0.5.0+pt28cpu', 'cpu': '11th Gen Intel(R) Core(TM) i5-11300H @ 3.10GHz', 'cpu_threads': 4, 'git_commit': '1fa59dfdf7d8f8ef4c283be98f4bb5e6e400bcf9', 'git_dirty': True}

| Model | Test ROC-AUC | Test AP | Test F1 | Val ROC-AUC | Train s | Test seeds/s | Params |
|---|---|---|---|---|---|---|---|
| HGT scratch, 100% labels | 0.702 ± 0.008 | 0.649 ± 0.011 | 0.676 ± 0.005 | 0.701 ± 0.009 | 22.8 | 2121 | 364807 |
| HGT scratch, 10% labels | 0.585 ± 0.077 | 0.543 ± 0.066 | 0.646 ± 0.003 | 0.569 ± 0.070 | 4.3 | 2104 | 364807 |
| Linear probe, random frozen HGT | 0.526 ± 0.008 | 0.492 ± 0.010 | 0.640 ± 0.001 | 0.523 ± 0.010 | 17.9 | 1940 | 65 |
| Linear probe, JEPA HGT | 0.633 ± 0.003 | 0.573 ± 0.006 | 0.645 ± 0.006 | 0.644 ± 0.009 | 10.0 | 2037 | 65 |
| JEPA HGT fine-tuned, 100% labels | 0.632 ± 0.006 | 0.570 ± 0.011 | 0.647 ± 0.008 | 0.667 ± 0.011 | 17.2 | 2194 | 364807 |
| JEPA HGT fine-tuned, 10% labels | 0.631 ± 0.021 | 0.582 ± 0.021 | 0.643 ± 0.008 | 0.656 ± 0.008 | 2.8 | 2188 | 364807 |

Mean ± std over seeds (model initialisation, batch order, neighbour sampling); the graph, the seeds and the splits are fixed. Thresholds for F1 are chosen on validation. Params: trainable parameters (LightGBM: number of trees).

## Paired comparisons (same test seeds)

| A | B | AUC(A) − AUC(B) | 95% paired bootstrap CI | P(diff ≤ 0) |
|---|---|---|---|---|
| Linear probe, JEPA HGT | Linear probe, random frozen HGT | +0.1069 | [+0.0748, +0.1394] | 0.000 |
| JEPA HGT fine-tuned, 100% labels | HGT scratch, 100% labels | -0.0703 | [-0.0877, -0.0517] | 1.000 |
| JEPA HGT fine-tuned, 10% labels | HGT scratch, 10% labels | +0.0458 | [+0.0219, +0.0690] | 0.000 |

Mean over model seeds of the per-seed AUC difference; 1,000 bootstrap resamples of the test seeds shared by both models.
