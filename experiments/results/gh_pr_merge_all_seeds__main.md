# gh_pr_merge_all_seeds__main

- config: `configs/gh_pr_merge_all_seeds.toml`
- seeds: [0, 1, 2]
- data: gharchive; seeds per split {'train': 91329, 'val': 28631, 'test': 25306}, positive rate {'train': 0.326, 'val': 0.311, 'test': 0.317}
- environment: {'backend': 'pyg', 'torch': '2.8.0+cpu', 'torch_geometric': '2.7.0', 'pyg_lib': '0.5.0+pt28cpu', 'cpu': '11th Gen Intel(R) Core(TM) i5-11300H @ 3.10GHz', 'cpu_threads': 4, 'git_commit': '84d6ec19a89c6103b68c3bbd12b84916cabc48a8', 'git_dirty': False}

| Model | Test ROC-AUC | Test AP | Test F1 | Val ROC-AUC | Train s | Test seeds/s | Params |
|---|---|---|---|---|---|---|---|
| LightGBM (+history) | 0.725 ± 0.000 | 0.516 ± 0.000 | 0.563 ± 0.001 | 0.677 ± 0.000 | 1.2 | 809037 | 82 |
| MLP (+history) | 0.726 ± 0.001 | 0.510 ± 0.003 | 0.566 ± 0.002 | 0.676 ± 0.000 | 23.1 | 34395 | 6657 |
| HeteroSAGE (+history) | 0.733 ± 0.001 | 0.520 ± 0.002 | 0.569 ± 0.001 | 0.682 ± 0.001 | 331.1 | 3034 | 531713 |

Mean ± std over seeds (model initialisation, batch order, neighbour sampling); the graph, the seeds and the splits are fixed. Thresholds for F1 are chosen on validation. Params: trainable parameters (LightGBM: number of trees).

## Paired comparisons (same test seeds)

| A | B | AUC(A) − AUC(B) | 95% paired bootstrap CI | P(diff ≤ 0) |
|---|---|---|---|---|
| HeteroSAGE (+history) | LightGBM (+history) | +0.0085 | [+0.0062, +0.0108] | 0.000 |

Mean over model seeds of the per-seed AUC difference; 1,000 bootstrap resamples of the test seeds shared by both models.
