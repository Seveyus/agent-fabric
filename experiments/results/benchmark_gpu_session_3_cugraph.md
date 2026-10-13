# Sampling / training-step throughput — backend `cugraph`

- host: {'platform': 'Linux-7.0.0-38-generic-x86_64-with-glibc2.39', 'python': '3.12.3', 'cpu': '11th Gen Intel(R) Core(TM) i5-11300H @ 3.10GHz'}
- backend: {'backend': 'cugraph', 'sampling': 'pylibcugraph on GPU (temporal: monotonically_decreasing)', 'feature_gather': 'WholeGraph FeatureStore', 'model_device': 'cuda', 'torch': '2.8.0+cu129', 'cugraph_pyg': '26.10.00', 'gpu': 'NVIDIA GeForce GTX 1650 with Max-Q Design'}

| Graph | Nodes | Edges | Graph MB | Fanout | Batch | Temporal | Sampling seeds/s | Sampled edges/s | Nodes/batch | SAGE train seeds/s (sampling share) | HGT train seeds/s (sampling share) | Invalid edges dropped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| task:task_pr_merge_48h.pt | 1,387,329 | 12,353,872 | 396 | [5, 3] | 256 | True | 1659 | 34481 | 5577 | 1395 (90%) | 1379 (87%) | 10 |
| task:task_pr_merge_48h.pt | 1,387,329 | 12,353,872 | 396 | [5, 3] | 1024 | True | 5057 | 95411 | 20344 | 4515 (86%) | 3924 (74%) | 16 |
| task:task_pr_merge_48h.pt | 1,387,329 | 12,353,872 | 396 | [10, 5] | 256 | True | 1489 | 40811 | 7275 | 1282 (91%) | 1259 (86%) | 10 |
| task:task_pr_merge_48h.pt | 1,387,329 | 12,353,872 | 396 | [10, 5] | 1024 | True | 4468 | 111087 | 26483 | 3679 (85%) | 3125 (74%) | 12 |
| task:task_pr_merge_48h.pt | 1,387,329 | 12,353,872 | 396 | [20, 10] | 256 | True | 1250 | 48903 | 10274 | 1102 (91%) | 1069 (84%) | 2 |
| task:task_pr_merge_48h.pt | 1,387,329 | 12,353,872 | 396 | [20, 10] | 1024 | True | 3696 | 128101 | 36512 | 3099 (84%) | 2558 (68%) | 5 |
| task:task_pr_merge_48h.pt | 1,387,329 | 12,353,872 | 396 | [10, 5] | 256 | False | 4652 | 323997 | 10112 | — | — | 0 |
| synthetic:2000_users | 12,910 | 217,026 | 6 | [5, 3] | 256 | True | 5560 | 77232 | 3678 | 4025 (72%) | 3819 (70%) | 0 |
| synthetic:2000_users | 12,910 | 217,026 | 6 | [5, 3] | 1024 | True | 20021 | 101179 | 5655 | 14061 (71%) | 13025 (67%) | 1 |
| synthetic:2000_users | 12,910 | 217,026 | 6 | [10, 5] | 256 | True | 5397 | 87989 | 4274 | 3773 (72%) | 3508 (69%) | 0 |
| synthetic:2000_users | 12,910 | 217,026 | 6 | [10, 5] | 1024 | True | 18479 | 100428 | 6010 | 13769 (73%) | 12202 (66%) | 0 |
| synthetic:2000_users | 12,910 | 217,026 | 6 | [20, 10] | 256 | True | 5233 | 95924 | 4775 | 3747 (72%) | 3447 (67%) | 0 |
| synthetic:2000_users | 12,910 | 217,026 | 6 | [20, 10] | 1024 | True | 18096 | 104467 | 6326 | 12553 (75%) | 11865 (67%) | 0 |
| synthetic:2000_users | 12,910 | 217,026 | 6 | [10, 5] | 256 | False | 8484 | 684378 | 6953 | — | — | 0 |
| synthetic:20000_users | 126,578 | 2,153,982 | 63 | [5, 3] | 256 | True | 5598 | 136721 | 6509 | 4034 (71%) | 3495 (63%) | 13 |
| synthetic:20000_users | 126,578 | 2,153,982 | 63 | [5, 3] | 1024 | True | 16556 | 320351 | 20839 | 10595 (67%) | 7881 (50%) | 27 |
| synthetic:20000_users | 126,578 | 2,153,982 | 63 | [10, 5] | 256 | True | 5061 | 170506 | 8883 | 3717 (74%) | 2993 (61%) | 24 |
| synthetic:20000_users | 126,578 | 2,153,982 | 63 | [10, 5] | 1024 | True | 13701 | 351615 | 27311 | 8702 (66%) | 6282 (47%) | 136 |
| synthetic:20000_users | 126,578 | 2,153,982 | 63 | [20, 10] | 256 | True | 4197 | 194323 | 12111 | 3101 (74%) | 2531 (59%) | 36 |
| synthetic:20000_users | 126,578 | 2,153,982 | 63 | [20, 10] | 1024 | True | 12727 | 393862 | 32716 | 7828 (64%) | 5525 (45%) | 62 |
| synthetic:20000_users | 126,578 | 2,153,982 | 63 | [10, 5] | 256 | False | 8090 | 865283 | 14224 | — | — | 0 |
