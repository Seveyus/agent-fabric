# Graph learning in Agent Fabric — design notes

Audience: engineers who build or review graph-ML systems (PyG, cuGraph-PyG).
This document explains what is implemented, why it is built this way, what was
measured, and what is not verified. Numbers quoted here are copied from the
files in [`experiments/results/`](../experiments/results/); nothing is
extrapolated.

- [1. Status](#1-status)
- [2. Problem and why a graph](#2-problem-and-why-a-graph)
- [3. Data](#3-data)
- [4. Temporal heterogeneous graph](#4-temporal-heterogeneous-graph)
- [5. Leakage control](#5-leakage-control)
- [6. Models](#6-models)
- [7. Self-supervised objective: temporal node JEPA](#7-self-supervised-objective-temporal-node-jepa)
- [8. Sampling backends: PyG and cuGraph-PyG](#8-sampling-backends-pyg-and-cugraph-pyg)
- [9. Evaluation protocol](#9-evaluation-protocol)
- [10. Results](#10-results)
- [11. Performance: where the time goes](#11-performance-where-the-time-goes)
- [12. Limitations and next steps](#12-limitations-and-next-steps)

## 1. Status

| Component | Status | Where |
|---|---|---|
| Streaming GH Archive ingest → typed Parquet | IMPLEMENTED, tested | `core/dataset/gharchive_ingest.py` |
| Temporal heterogeneous PR graph + PR-merge task | IMPLEMENTED, tested | `core/dataset/temporal_graph.py` |
| Point-in-time features, embargoed splits | IMPLEMENTED, tested against brute force | same |
| Temporal neighbour sampling, PyG + pyg-lib (CPU) | IMPLEMENTED, tested (no future edges) | `training/backends.py` |
| Temporal neighbour sampling, cuGraph-PyG (GPU) | **NOT VERIFIED** — written against the 26.10 API, never executed (no usable GPU driver here) | `training/backends.py` |
| MLP / HeteroSAGE / HGT node classifiers | IMPLEMENTED, tested | `models/hetero_gnn.py` |
| Temporal node JEPA pretraining | IMPLEMENTED, tested; downstream benefit **not shown** | `models/temporal_jepa.py`, `training/pretrain_jepa.py` |
| Config-driven runs, checkpoints, multi-seed suites | IMPLEMENTED | `training/node_task.py`, `training/suite.py` |
| Throughput benchmark | IMPLEMENTED; CPU results measured, GPU NOT RUN | `benchmarks/sampling_throughput.py` |
| EV fleet example (synthetic) | IMPLEMENTED | `core/dataset/ev_fleet.py` |
| Snapshot-level JEPA (original prototype) | Fixed to run and to be leakage-free; not evaluated on real data | `models/world_model.py`, `training/train.py` |
| Pre-flight API using a learned model | PLANNED (the API still uses heuristic scores) | `apps/api/routers/preflight.py` |

## 2. Problem and why a graph

Software delivery is relational: whether a change lands depends on who wrote
it, who reviews in that repository, how that repository has treated similar
changes, and how the people involved behaved elsewhere. The concrete,
measurable task used here is:

> At the moment a pull request is opened, predict whether it will be merged
> within 48 hours.

It is a proxy for delivery friction (PRs that stall), it has unambiguous
labels in public data, and its answer plausibly depends on context outside the
PR itself: the author's track record, the repository's merge behaviour, the
reviewers' activity. That last point is a hypothesis to test, not an
assumption: hand-built history aggregates may capture the same context, and
the experiments compare against exactly that (§10).

## 3. Data

**Source.** [GH Archive](https://www.gharchive.org/) hourly dumps of public
GitHub events. `core/dataset/gharchive_ingest.py` streams each hour over HTTP,
decompresses multi-member gzip on the fly and keeps only the fields the graph
needs (PR open-time payload, close/merge outcome, reviews, comments,
hourly-aggregated pushes, stars, forks, issues, releases). Raw JSON never hits
the disk: ~120 MB compressed per hour → ~2 MB Parquet.

**Payload trimming (measured).** GitHub trimmed public Events API payloads
between 2025-10-06 and 2025-10-13. After that, `PullRequestEvent` only carries
`{base, head, id, number, url}` — no title, author, merge flag or size — and
`PushEvent` has no commit list. A 2026-10-01 hour had 63 PR events in its
first 30k lines and none with a title; 2025-10-06 had 2,865, all with titles.
Any PR-level task must use data up to 2025-10-05. (The repository's original
`validate_signal.py` run used May 2026 data, so its PR features were all zero.)

**Collection used for the results.** 2025-08-25 → 2025-09-15 (UTC, all 24
hours, 528 files). To bound disk usage, collection is two-phase: the warm-up
week is ingested fully, then only repositories with ≥ 2 distinct non-bot PR
authors during the warm-up are kept (24,475 repositories), including for the
following weeks. This removes single-owner repositories where "merge" is a
self-click, and it means cross-repository context is limited to those
24,475 repositories. Bot accounts (`[bot]` suffix, `-bot` names, known
apps) are kept in the graph but excluded from the seeds.

## 4. Temporal heterogeneous graph

One `HeteroData` holds the whole period. Every edge carries `time` (unix
seconds) = when the interaction happened.

| Node type | Features |
|---|---|
| `pr` | open-time payload: size (log additions/deletions/files/commits), draft, title/body length, author association (7 one-hot), targets default branch, labels, requested reviewers, revert, author is bot, hour/weekday; plus 10 **point-in-time history aggregates** (author's prior opened / merged / closed PRs and smoothed merge rate, prior reviews given, prior merges in this repo; repo's prior opened / merged / closed and merge rate), all computed strictly before the open time |
| `user` | `[1, is_bot]` |
| `repo` | `[1]` |

| Relation (each with a `rev_*` twin) | Time |
|---|---|
| user –opens→ pr | open |
| pr –in_repo→ repo | first seen |
| pr –merged_into→ repo, pr –closed_unmerged_in→ repo | outcome |
| user –merges→ pr, user –closes→ pr | outcome (actor) |
| user –authored_merged→ pr, user –authored_closed→ pr | outcome (the PR's author) |
| user –approves→ pr, user –reviews→ pr, user –comments_on→ pr | interaction |
| user –pushes_to / stars / forks / issue_activity / releases→ repo | interaction |

Design decisions:

* **User and repo nodes are featureless on purpose.** Their state is
  time-dependent (a repo's merge rate changes daily); a static feature would
  either leak the future or be stale. Their state is reconstructed from the
  time-filtered neighbourhood instead. Learnable per-node embeddings were not
  used: ~0.5M users, most seen a handful of times, and the model must be
  inductive for new users.
* **Author-outcome relations** (`authored_merged`, `authored_closed`) exist so
  that an author's past outcomes are two hops from a new PR of that author
  (seed ← author ← earlier PR). Through `opens` and `merges` only, it takes
  three hops (seed ← author ← earlier PR ← merger), out of reach of a 2-layer
  model. On the synthetic benchmark this moved HGT from 0.58 to 0.70 test
  ROC-AUC without history features (single seed, before the HGTConv fix in §6).
* **Reverse relations are explicit**, so every node type receives messages.

Scale of the graph used for the results (from
`experiments/results/gh_pr_merge_48h__main.md`): see §10.

## 5. Leakage control

Three mechanisms, each with a test in `tests/ml/`:

1. **Temporal neighbour sampling.** For a seed PR opened at *t*, only edges
   with `time ≤ t` are sampled. Measured on PyG 2.7 + pyg-lib 0.5 with a toy
   graph: an edge at exactly the seed time is included, the constraint applies
   at every hop relative to the *seed* time (not path-monotone), and
   `disjoint=True` is forced. Consequence: the seed's own `opens` / `in_repo`
   edges are visible, its merge, close, review and comment edges are not.
   `test_pyg_temporal_sampling_never_returns_future_edges` checks every
   sampled edge against its own seed's time;
   `test_seed_own_outcome_edges_are_invisible` checks that no outcome relation
   ever involves the seed itself.
2. **Point-in-time node features.** PR features are frozen at open time; a PR
   reached as a neighbour was opened before the seed (any edge to it is ≥ its
   open time), so its features are known at the seed time. History aggregates
   use `join_asof(..., allow_exact_matches=False)`; checked against a brute
   force recount in `test_history_features_are_point_in_time`.
3. **Embargoed temporal splits.** Train / validation / test are defined by open
   time and each split's labels must be resolved before the next split starts
   (`train_end + 48h ≤ val_start`, `val_end + 48h ≤ test_start`, enforced by
   `PRMergeTaskConfig.validate`). Without the embargo, weights are fit on
   outcomes that happen after some evaluation PRs were opened.

Normalisation statistics for PR features come from the training seeds only.

## 6. Models

All encoders implement `encoder(batch, seed_type) -> (batch_size, hidden)` and
share the loader, loss (BCE), optimiser (AdamW), early stopping (validation
ROC-AUC) and threshold selection, so differences are attributable to the model.

* **MLP** — seed features only, sampled with fan-out 0. The non-graph control
  inside the same training loop.
* **HeteroSAGE** — one `SAGEConv` per relation, **sum** aggregation within and
  across relations, residual + LayerNorm. Sum rather than mean so that counts
  (how many of an author's past PRs were merged vs. closed, capped by the
  fan-out) survive aggregation; with mean aggregation per relation only the
  presence of a relation is visible.
* **HGT** — PyG `HGTConv` (type-specific K/Q/V, relation-specific attention and
  message matrices, learned gated skip), 2 layers, 2 heads, LayerNorm. PyG's
  implementation has no relative temporal encoding (RTE) from the HGT paper;
  time enters only through sampling.

**Why HGT here.** The graph has 3 node types and 32 relation types whose
semantics differ sharply (an `authored_closed` edge means the opposite of
`authored_merged`). HGT's per-relation attention lets the softmax over a
node's incoming edges mix relation types, so the *proportion* of merged vs.
closed past PRs can influence the representation. HeteroSAGE with sum
aggregation is the cheaper baseline that encodes counts directly. Whether
either beats engineered aggregates is an empirical question (§10).

**HGTConv pitfall found during this work.** `HGTConv` flattens relation types
in the iteration order of `edge_index_dict` and passes them to
`HeteroLinear(..., is_sorted=True)`. When that order differs from the
`metadata` order — the normal case when forward/reverse relations are
inserted pairwise — relation-specific K/V weights are applied to the wrong
node segments **silently** on the naive path, and `segment_matmul` crashes
with negative split sizes once the timing cache selects it (that crash is how
it surfaced). Same code on PyG master (checked 2026-10-12). Both HGT wrappers
now pass edges in metadata order;
`test_pyg_hgtconv_is_order_sensitive_and_our_wrapper_is_not` documents the
upstream behaviour and guards the workaround. It has not been reported
upstream yet.

## 7. Self-supervised objective: temporal node JEPA

**What is predicted.** For an anchor PR *p* opened at *t*:

```
context  x = neighbourhood of p at t            (edges with time ≤ t)          -> online encoder f_θ
target   y = interactions of p in (t, t+Δ]      (p's features zeroed,
             edges ≤ t into p removed)                                          -> target encoder f_ξ (EMA of θ)
s_x = f_θ(x)[p]      s_y = LayerNorm(f_ξ(y)[p])   (stop-gradient)
ŝ_y = g_φ(s_x, MLP(log(1+Δ)))                      Δ ∈ {12, 24, 48} h, sampled per anchor
L   = ‖ŝ_y − s_y‖²      ξ ← τ ξ + (1−τ) θ,  τ: 0.996 → 1 (cosine)
```

The model forecasts, in latent space, what will happen to a PR in the next Δ
hours (who reviews, comments, merges or closes it) from what is known when it
is opened. No labels are used.

**Why a future-only target.** If the target were the full neighbourhood at
*t+Δ*, it would contain the context; an encoder that ignores new edges makes
the two views identical and the loss zero. A temporal BYOL therefore learns
invariance to time — the opposite of anticipating change. Masking the
anchor's features and its past edges makes context and target carry disjoint
information, the way I-JEPA predicts target blocks from a disjoint context
block; Δ plays the role of the target position.

**What is verified.** Stop-gradient (no gradient reaches ξ), EMA update and
schedule, context/target batch alignment, the target view contains only
edges in (t, t+Δ] into the anchor (`tests/ml/test_temporal_jepa.py`), and
anchors end before validation starts (`t + max Δ ≤ val_start`; unlabelled
warm-up PRs are anchors too).

**What was found** (details §10.3):

* The target view does encode the future: a linear probe on target-encoder
  embeddings of (t, t+48h] separates merged / not merged with AUC 1.00 on
  synthetic data (random or trained encoder); the merge label explains 28 %
  of the target variance.
* The objective learns slowly: the loss normalised by target variance stays at
  1.00 (predicting the mean target) for ~10 epochs, then falls to ~0.93 after
  40. With 8 epochs the representation is no better than a random encoder.
* After 40 epochs the frozen JEPA encoder beats a frozen random encoder under a
  linear probe by +0.107 AUC, and fine-tuning it with 10 % of the labels beats
  training from scratch on those labels by +0.046 (paired CIs exclude 0).
* Fine-tuning with the from-scratch optimiser settings stays near the probe
  level (~0.63), so with all labels it is 0.07 *worse* than training from
  scratch (0.70): negative transfer. Lower encoder learning rates or longer
  patience were not tried.

So the objective is meaningful and the machinery is verified, but in its
current form the pretrained encoder is only useful in the low-label regime.

The original snapshot-level JEPA (`models/world_model.py`, per-repo weekly
windows) is kept and was fixed (it could not import or run, and its predictor
was given the event summary of the target window). It is a different
granularity — repository state, not PR — and is not evaluated here.

## 8. Sampling backends: PyG and cuGraph-PyG

`training/backends.py` exposes one loader interface. Both backends produce a
`HeteroData` with `batch[seed].input_id`, `.batch_size`, `.n_id` and the seeds
first in input order — verified for PyG; for cuGraph-PyG this is what its
26.10 sampler code sets (`cugraph_pyg/sampler/sampler.py`). The model and
training loop do not know which backend produced the batch.

| | `pyg` | `cugraph` |
|---|---|---|
| Graph storage | `HeteroData` in host memory; per-relation CSC built by PyG | cuGraph graph in GPU memory (`cugraph_pyg.data.GraphStore`) |
| Features, edge times | host tensors | WholeGraph `FeatureStore` (`feature_store[edge_type, "time", None]` for temporal sampling, required for every edge type) |
| Neighbour sampling | pyg-lib, C++ on CPU threads | pylibcugraph on the GPU |
| Temporal rule | every sampled edge `time ≤ seed_time` (all hops vs. the seed) | `temporal_comparison="monotonically_decreasing"`: times must not increase along the sampled path, starting from the seed time — a subset of PyG's candidates |
| Output | `HeteroData` (host), then `.to(device)` | `HeteroData` produced on the GPU |
| Model forward/backward | PyTorch | PyTorch |
| Init | none | `torch.distributed` (NCCL) → cuGraph comms → RMM → WholeGraph, as in the cuGraph-PyG examples |
| Status here | measured | **NOT VERIFIED** (never executed) |

Consequences for a reviewer:

* The two backends are **not** expected to produce identical batches (different
  temporal rule, different RNG); model quality, not batches, should be
  compared across them.
* Temporal JEPA pretraining needs per-edge timestamps in the sampled batch to
  build the future-only view; this is wired for PyG only. The cuGraph-PyG
  equivalent would be fixed-window sampling (`input_start_time` /
  `input_end_time`), not implemented.
* HGT on cuGraph-PyG batches: the batches are ordinary `HeteroData`, so
  `HGTConv` should consume them (with the edge-order fix above). The upstream
  request for heterogeneous node-classification / HGT examples
  ([rapidsai/cugraph-gnn#365](https://github.com/rapidsai/cugraph-gnn/issues/365))
  was closed as "not planned" on 2026-10-01, so this combination has no
  upstream reference either.

**Running the GPU path** (requires Linux, NVIDIA GPU with compute capability
≥ 7.0, CUDA 12 driver ≥ 525.60.13):

```bash
pip install -r requirements/graph-cugraph-cu12.txt
pytest tests/ml -m cugraph                                  # contract + no-future-edge test
python -m benchmarks.sampling_throughput --backend cugraph --device cuda --scales 2000 20000 60000
python -m training.node_task --config configs/gh_pr_merge.toml --set backend.name='"cugraph"' --set backend.device='"cuda"'
```

On the development laptop (GTX 1650, CC 7.5, 4 GB) this is blocked by the
driver: the NVIDIA module exists only for kernel 7.0.0-31 while 7.0.0-34 runs.

## 9. Evaluation protocol

* Same seeds, splits and features for every model; seeds subsampled
  deterministically (`max_train_seeds`, `max_eval_seeds`, `sample_seed`).
* Early stopping and the F1 threshold use the validation split only; test is
  scored once, from the best validation checkpoint.
* Reported: ROC-AUC (with 95% bootstrap CI per run), average precision, F1 /
  precision / recall at the validation threshold, training time, inference
  throughput (sampling + forward), parameter count. Mean ± std over 3 seeds
  (initialisation, batch order, sampling); data and splits fixed.
* 2 × 2 design: {non-graph, graph} × {intrinsic features, + point-in-time
  history aggregates}. The question for the GNN is whether learning from the
  raw temporal graph adds signal beyond hand-built aggregates — not whether it
  beats a deliberately weak baseline.
* Timing columns of the suites are indicative (other jobs could share the
  CPU); throughput claims come from the benchmark run in isolation (§11).

### 9.1 Reproducibility controls (found and fixed during this work)

Three sources of run-to-run variation with a fixed seed were measured and
removed; each has a test.

| Problem | Evidence | Fix |
|---|---|---|
| Graph build depended on polars tie-breaking: edges de-duplicated then sorted by timestamp only, so same-second edges could be stored in a different order on every rebuild, changing the CSC neighbour order and therefore what a seeded sampler returns. | Two in-process runs with the same seed: different epoch-1 losses. | Sort edges by (time, src, dst). `test_build_is_deterministic` rebuilds from shuffled events and compares tensors. |
| pyg-lib's multi-threaded disjoint sampling ignores `torch.manual_seed` (single-threaded it is deterministic). | Same seed, same seed order, different sampled users. | `backend.num_workers = 1` (default): DataLoader workers sample single-threaded with a deterministic per-worker seed while the model keeps all threads. Cost on this 4-core CPU: ~1.6× slower HeteroSAGE epochs (the worker competes for cores). `test_same_seed_gives_identical_results`. |
| PyG `HeteroLinear` (HGTConv's relation-specific K/V) chooses `segment_matmul` vs. a per-type loop from a one-shot timing per module; under load it sometimes chose the loop. | 9.3 s vs 0.76 s per HGT train step on the GH graph (batch 512, fan-out [10, 5]). | `backend.segment_matmul = "on"` pins pyg-lib's kernel. |

Suites also run each (variant, seed) in a fresh process: inside one process,
memory grew from 3.1 to 6.1 GB across runs (task reloads, forked sampling
workers) and later runs slowed down.

## 10. Results

### 10.1 Real data: GH Archive, 48 h PR merge

Source: [`experiments/results/gh_pr_merge__main.md`](../experiments/results/gh_pr_merge__main.md)
(code `b0653b7`; the working tree differed only by README edits and a blank
line in a test fixture). Graph: 815,976 PRs, 546,878 users, 24,475 repos,
12,353,872 directed edges, built from 4,562,538 events. Seeds: 40,000 train /
15,000 validation / 15,000 test PRs (of 91,329 / 28,631 / 25,306 eligible),
32 % merged within 48 h. Fan-out [10, 5] per relation, batch 512, 2 layers,
hidden 64. Mean ± std over 3 seeds.

| Model | Test ROC-AUC, intrinsic | Test ROC-AUC, + history | Test AP, + history | Train time (s) | Test seeds/s (CPU) |
|---|---|---|---|---|---|
| Logistic regression | 0.648 ± 0.000 | 0.718 ± 0.000 | 0.514 | 0.7 | ~3.8 M |
| LightGBM | 0.681 ± 0.000 | 0.722 ± 0.001 | 0.520 | 0.4 | ~1.1 M |
| MLP (same loop as GNNs) | 0.678 ± 0.001 | 0.724 ± 0.001 | 0.512 | 5 | ~83 k |
| HeteroSAGE | **0.721 ± 0.002** | **0.730 ± 0.001** | 0.520 | 87 | ~5.6 k |
| HGT | 0.715 ± 0.002 | 0.728 ± 0.001 | 0.523 | 166 | ~3.0 k |

Paired bootstrap on the same 15,000 test PRs (AUC difference, 95 % CI):

| Comparison | Δ ROC-AUC | 95 % CI |
|---|---|---|
| HeteroSAGE vs MLP, intrinsic features | +0.043 | [+0.037, +0.049] |
| LightGBM + history vs LightGBM intrinsic | +0.042 | [+0.035, +0.048] |
| HeteroSAGE + history vs LightGBM + history | +0.0075 | [+0.0038, +0.0111] |
| HGT + history vs LightGBM + history | +0.0052 | [+0.0016, +0.0091] |
| HGT vs HeteroSAGE, + history | −0.0023 | [−0.0043, −0.0003] |

Reading:

1. **The temporal graph recovers the history signal.** Without engineered
   history features, message passing over time-filtered neighbourhoods adds
   +0.04 AUC over the same features without message passing — about the same
   gain that point-in-time author/repo aggregates give a gradient-boosted
   model.
2. **On top of those aggregates the graph adds little, but not nothing.**
   +0.005 to +0.0075 AUC, statistically clear on paired tests, small in
   practice.
3. **HGT does not earn its cost here.** It is slightly worse than
   HeteroSAGE with sum aggregation and trains ~1.9× slower; per-relation
   attention does not help on this schema at this scale.
4. **Cost.** On this CPU the best GNN trains ~200× longer than LightGBM and
   scores ~190× fewer PRs per second (5.6 k vs ~1.1 M; GNN figure includes
   neighbour sampling). For this task, a GBDT on point-in-time aggregates is
   the sensible production default; the GNN is worth it only if its cost
   drops (GPU) or if the aggregates are hard to maintain.
5. Validation AUCs are ~0.05 lower than test AUCs for every model; the
   validation PRs were opened on Monday–Tuesday, the test PRs on
   Friday–Saturday. Model ranking is the same on both.

Sampling vs compute on CPU (from the same runs, HeteroSAGE + history): an
epoch over 40,000 seeds spends 1.2 s waiting for the sampler and 10.2 s in
forward/backward; at inference 0.5 s sampling vs 2.1 s forward for 15,000
seeds (17.3 k nodes and 25.6 k edges per 512-seed batch). HGT: 1.2 s vs
24.2 s per epoch.

### 10.2 Synthetic: does the pipeline recover a known relational signal?

Source: [`experiments/results/synthetic_pr_merge__main.md`](../experiments/results/synthetic_pr_merge__main.md).
Synthetic events (`core/dataset/synthetic_events.py`) where merges are driven
by a latent author skill and repository strictness that are only observable
through earlier outcomes. 2,982 / 957 / 1,461 seeds, ~47 % positive.

| Model | Test ROC-AUC, intrinsic | + history |
|---|---|---|
| Logistic regression | 0.533 | **0.739** |
| LightGBM | 0.514 | 0.714 |
| MLP | 0.526 | 0.728 |
| HeteroSAGE | 0.692 | 0.730 |
| HGT | **0.702** | 0.727 |

Without history features the non-graph models are at chance, as designed, and
the temporal GNNs recover most of the signal (HGT vs MLP +0.176, paired CI
[0.145, 0.207]). With history features every model is close; logistic
regression is best because the planted mechanism is itself logistic in the
quantities the aggregates estimate. This is an end-to-end check of the
sampler, schema (author-outcome relations), models and loop — not evidence
about real data.

### 10.3 Temporal JEPA

Synthetic PR graph, intrinsic features only (history must come from the
graph), 3 seeds, 40 pretraining epochs on 4,989 anchors (unlabelled warm-up
PRs included). Source: [`experiments/results/synthetic_pr_merge__jepa.md`](../experiments/results/synthetic_pr_merge__jepa.md).

| Variant | Test ROC-AUC |
|---|---|
| HGT from scratch, 100 % labels | 0.702 ± 0.008 |
| HGT from scratch, 10 % labels | 0.585 ± 0.077 |
| Linear probe on a random frozen HGT | 0.526 ± 0.008 |
| Linear probe on the JEPA HGT | 0.633 ± 0.003 |
| JEPA HGT fine-tuned, 100 % labels | 0.632 ± 0.006 |
| JEPA HGT fine-tuned, 10 % labels | 0.631 ± 0.021 |

| Paired comparison | Δ ROC-AUC | 95 % CI |
|---|---|---|
| JEPA probe vs random probe | +0.107 | [+0.075, +0.139] |
| JEPA fine-tuned vs scratch, 100 % labels | −0.070 | [−0.088, −0.052] |
| JEPA fine-tuned vs scratch, 10 % labels | +0.046 | [+0.022, +0.069] |


## 11. Performance: where the time goes

Source: [`experiments/results/benchmark_pyg_cpu.md`](../experiments/results/benchmark_pyg_cpu.md)
— run alone on the machine, i5-11300H, 4 threads, PyG 2.7 + pyg-lib 0.5,
`segment_matmul` pinned, sampling in the main process (multi-threaded).

**Measured (CPU).**

| Graph | Edges | Temporal sampling, seeds/s | Sampled edges/s | HeteroSAGE train step, seeds/s | HGT train step, seeds/s | Sampling share of a train step |
|---|---|---|---|---|---|---|
| GH Archive (1.39 M nodes, 396 MB of tensors) | 12.4 M | 42 k – 90 k | 1.6 M – 4.7 M | 2.6 k – 5.9 k | 1.3 k – 2.6 k | 2 – 10 % |
| synthetic, 2 k users | 0.22 M | 59 k – 149 k | 2.5 M – 8.8 M | 2.2 k – 4.8 k | 1.1 k – 2.1 k | 1 – 7 % |
| synthetic, 60 k users | 6.5 M | 45 k – 96 k | 1.8 M – 5.4 M | 2.1 k – 4.6 k | 1.0 k – 2.1 k | 1 – 8 % |

Ranges span fan-outs [5,3] / [10,5] / [20,10] and batch sizes 256 / 1024.
Larger fan-outs and smaller batches lower sampling throughput; at [20, 10] and
batch 1024 a GH batch has 37 k nodes. Non-temporal (non-disjoint) sampling at
[10, 5] / 256 was *slower* than temporal (36 k vs 48 k seeds/s on GH) because
it returns larger neighbourhoods (8.9 k vs 6.2 k nodes per batch).

**Answers to the questions a RAPIDS engineer would ask** (measured where
stated, otherwise reasoning):

1. *What would cuGraph-PyG accelerate here?* Neighbour sampling (temporal,
   heterogeneous, disjoint) and the feature/time gather for sampled nodes:
   `GraphStore` builds the cuGraph graph in GPU memory, `FeatureStore` holds
   features and edge times in WholeGraph memory, and the loader returns
   GPU-resident `HeteroData`. Not touched: graph construction from events
   (polars on CPU; cuDF would be the RAPIDS equivalent, not implemented),
   tabular baselines (cuML / GPU XGBoost not used).
2. *What stays in PyTorch?* Model forward/backward (HGTConv / SAGEConv /
   MLP), loss, optimiser, metrics — on whatever device the model is on.
3. *What runs on the GPU?* With `--backend cugraph`: sampling, gather and the
   model. With `--backend pyg --device cuda`: only the model; sampling stays on
   CPU and each batch is copied to the GPU.
4. *Memory and transfer.* The GH graph is 396 MB of tensors (edge indices,
   edge times, features). One [10, 5] batch of 512 seeds is ~17 k nodes and
   ~26 k edges, i.e. a few MB of features and indices — a small host→device
   copy per batch on the PyG path. The cuGraph path keeps the graph on the GPU,
   so GPU memory must hold the cuGraph graph (both directions are explicit,
   12.4 M edges), edge times (cuGraph converts them to double for `last`
   sampling), WholeGraph features, RMM pools and the model; on a 4 GB GPU this
   is plausible for this graph but unmeasured.
5. *How does sampling affect performance?* On this CPU it barely does:
   2-10 % of a train step (measured). The model compute dominates. Therefore
   a GPU sampler alone would not speed this pipeline up on CPU — the gain
   requires the model on a GPU too.
6. *When could GPU sampling matter?* (reasoning, not measured) Once the model
   runs on a GPU, each train step gets much faster while the CPU sampler stays
   at ~42-90 k seeds/s on this graph. If a GPU HGT step reached tens of
   thousands of seeds/s, CPU sampling would become the bottleneck and moving
   it to the GPU would be the next step. Separately, the host-memory path does
   not scale to the full GitHub event graph (this graph is 24 k of the
   millions of active repositories); multi-GPU storage (WholeGraph) and
   sampling is the relevant regime there. Both need measurement.
7. *Limitations of this implementation.* Not executed on a GPU; temporal rule
   differs from PyG (`monotonically_decreasing` path constraint vs. "≤ seed
   time" at every hop); JEPA pretraining not wired for cuGraph; HGT on
   cuGraph-PyG batches has no upstream example.

**Timing hygiene learned on the way** (§9.1): pin `segment_matmul`
(12× swing on HGT steps otherwise), isolate runs in processes, and remember
that reproducible sampling via a single DataLoader worker costs ~1.6× epoch
time on 4 cores.

## 12. Limitations and next steps

**Data and task**

* One collection window (3 weeks of September 2025) and one temporal test
  window. Stability over time (other months, other horizons) is not measured.
* The graph only contains the 24,475 repositories selected on warm-up
  activity; cross-repository context outside that set is invisible, and
  repositories that became active later are absent.
* Labels come from GH Archive events; a missing close event becomes a false
  negative. Bot detection is a login heuristic.
* Seeds are subsampled (40k of 91k training PRs, 15k of ~25-29k evaluation
  PRs) to keep CPU runs short.
* "Merged within 48 h" is a proxy for delivery friction, not a measure of
  delivery failure (reverts, incidents). Revert detection exists in the
  ingester (`is_revert`, `reverts_number`) but no revert task is evaluated:
  positives are too rare in three weeks.

**Models and training**

* No hyper-parameter search beyond defaults for any model; LightGBM uses fixed
  parameters with early stopping. A tuned GNN could do better; so could a
  tuned GBDT.
* HGT without relative temporal encoding; time is only a sampling cut-off, so
  a review from five minutes ago and one from three weeks ago look alike.
* User and repository nodes have no features of their own; their state is
  reconstructed from a fan-out-capped sample, which loses exact counts that
  the engineered aggregates have.
* Probabilities are not calibrated.

**Infrastructure**

* Every measurement is CPU-only. The cuGraph-PyG backend has not been executed;
  its temporal rule differs from PyG's, so batches are not comparable.
* Temporal JEPA pretraining is PyG-only (needs edge times in the batch).
* Reproducible sampling (`num_workers = 1`) costs ~1.6× epoch time on 4 cores.

**Before production use** (e.g. behind the pre-flight API): longer and
repeated temporal back-tests, calibration, per-prediction explanations,
monitoring of drift in repository mix and bot traffic, a decision on
whether the GNN's cost is justified over a GBDT on engineered aggregates (see
§10), and — for large organisations' graphs — the GPU sampling benchmark.
