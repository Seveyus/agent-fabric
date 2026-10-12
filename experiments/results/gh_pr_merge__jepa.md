# gh_pr_merge__jepa

- config: `configs/gh_pr_merge.toml`
- seeds: [0, 1, 2]
- data: gharchive; seeds per split {'train': 40000, 'val': 15000, 'test': 15000}, positive rate {'train': 0.324, 'val': 0.311, 'test': 0.32}
- environment: {'backend': 'pyg', 'torch': '2.8.0+cpu', 'torch_geometric': '2.7.0', 'pyg_lib': '0.5.0+pt28cpu', 'cpu': '11th Gen Intel(R) Core(TM) i5-11300H @ 3.10GHz', 'cpu_threads': 4, 'git_commit': '84d6ec19a89c6103b68c3bbd12b84916cabc48a8', 'git_dirty': False}

| Model | Test ROC-AUC | Test AP | Test F1 | Val ROC-AUC | Train s | Test seeds/s | Params |
|---|---|---|---|---|---|---|---|
| HGT scratch, 100% labels | 0.715 ± 0.002 | 0.499 ± 0.007 | 0.560 ± 0.001 | 0.673 ± 0.001 | 253.2 | 1905 | 364807 |
| HGT scratch, 10% labels | 0.691 ± 0.003 | 0.467 ± 0.005 | 0.548 ± 0.005 | 0.656 ± 0.001 | 43.6 | 1908 | 364807 |
| Linear probe, random frozen HGT | 0.651 ± 0.002 | 0.434 ± 0.003 | 0.520 ± 0.000 | 0.615 ± 0.002 | 114.8 | 2946 | 65 |
| Linear probe, JEPA HGT | 0.696 ± 0.005 | 0.486 ± 0.009 | 0.546 ± 0.005 | 0.659 ± 0.003 | 93.5 | 3178 | 65 |
| JEPA HGT fine-tuned, 100% labels | 0.717 ± 0.001 | 0.505 ± 0.003 | 0.561 ± 0.001 | 0.672 ± 0.002 | 164.6 | 3098 | 364807 |
| JEPA HGT fine-tuned, 10% labels | 0.695 ± 0.003 | 0.476 ± 0.008 | 0.547 ± 0.003 | 0.658 ± 0.003 | 19.4 | 3052 | 364807 |

Mean ± std over seeds (model initialisation, batch order, neighbour sampling); the graph, the seeds and the splits are fixed. Thresholds for F1 are chosen on validation. Params: trainable parameters (LightGBM: number of trees).

## Paired comparisons (same test seeds)

| A | B | AUC(A) − AUC(B) | 95% paired bootstrap CI | P(diff ≤ 0) |
|---|---|---|---|---|
| Linear probe, JEPA HGT | Linear probe, random frozen HGT | +0.0443 | [+0.0389, +0.0502] | 0.000 |
| JEPA HGT fine-tuned, 100% labels | HGT scratch, 100% labels | +0.0015 | [-0.0004, +0.0034] | 0.060 |
| JEPA HGT fine-tuned, 10% labels | HGT scratch, 10% labels | +0.0043 | [+0.0006, +0.0082] | 0.011 |

Mean over model seeds of the per-seed AUC difference; 1,000 bootstrap resamples of the test seeds shared by both models.
