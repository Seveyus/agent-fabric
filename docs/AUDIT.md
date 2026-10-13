# Repository audit (baseline: commit `b1ab303`, 2026-10-12)

Scope: the graph-learning part of Agent Fabric (`core/dataset/`, `models/`,
`training/`, `scripts/validate_signal.py`) and the README claims about it.
The FastAPI / worker / connector platform was read but not modified.

Each claim was checked by reading the code and, where possible, by running it.
Status labels: **IMPLEMENTED** (works as described), **PARTIAL** (exists but
broken or weaker than described), **NOT VERIFIED** (code exists, never run or no
evidence), **PLANNED** (described, not in code).

## Findings

| Area | Status at baseline | Evidence |
|---|---|---|
| GH Archive download + event normalisation (`core/dataset/gh_archive.py`) | IMPLEMENTED | Works, but keeps raw files and full dicts in memory. |
| GH Archive payload content | NOT VERIFIED → **broken for post-2025-10 data** | Measured: from 2025-10-13 on, `PullRequestEvent.payload.pull_request` only has `{base, head, id, number, url}`; `PushEvent` has no commits. Rich payloads end between 2025-10-06 and 2025-10-13. |
| `scripts/validate_signal.py` baseline | PARTIAL | Ran on May 2026 data, so commit counts and PR merges/titles are structurally zero. 42 windows, 4 positives; LogReg "AUC 1.0" was fit and scored on the same rows ("full_eval_mode"). Not evidence of signal. |
| Heterogeneous snapshot graph (`graph_builder.py`, `training/dataset.py`) | PARTIAL | All edges fully-connected bipartite (every PR ↔ every zone, every contributor ↔ every PR); file zones are guessed from titles. The graph carried no relational information. |
| HGT encoder (`models/encoder.py`) | PARTIAL (crashed) | No reverse edges → `contributor` never a destination → HGTConv drops it → `KeyError` at layer 2. |
| JEPA world model (`models/world_model.py`) | PARTIAL | Online/target encoders, EMA, predictor, stop-gradient: present. But the predictor was conditioned on the event summary *of the target window* (answer given as input); fixed EMA τ; "best" checkpoint chosen on a loss whose target moves. |
| Training script (`training/train.py`) | PARTIAL (could not import) | `training/dataset.py` imported `NODE_TYPES` from a module that does not define it; unseeded `random_split` across time (temporal leakage). |
| Evaluation script (`training/evaluate.py`) | PARTIAL (could not import) | Imported a non-existent `collate_fn`; scored the training pairs; rebuilt the model with default config regardless of checkpoint. |
| "Trained on millions of public GitHub project timelines" (README) | NOT VERIFIED (false) | No training run, checkpoint or result existed. |
| "~700K parameters" (README) | Incorrect | 4.07M trainable parameters with the default config (measured). |
| `pip install torch-scatter torch-sparse` (README) | Unnecessary | PyG ≥ 2.3 works without them; temporal sampling needs `pyg-lib`. |
| Compute table (README: "~8h on 4 GB", "<10 ms inference") | NOT VERIFIED | No measurement in the repo. |
| Pre-flight API probabilities (`apps/api/routers/preflight.py`) | Heuristic, not learned | `probability = stored confidence × hand-set multiplier per relation` (e.g. `blocks: 0.95`); no model is connected. |
| GPU support | PLANNED | `--device cuda` flag only; no RAPIDS / cuGraph code. |
| Tests | Minimal | 2 unit tests (chunking, connector discovery), none for the ML code; no CI. |

## Development environment (measured, relevant to what could be run)

* CPU: Intel i5-11300H (4 cores / 8 threads), 15 GB RAM, ~1.5 GB free disk.
* GPU: GeForce GTX 1650 Mobile (Turing, compute capability 7.5, 4 GB) is present
  but **unusable**: the NVIDIA 595 kernel module is built for kernel
  7.0.0-31 while 7.0.0-34 is running (`nvidia-smi` fails). CUDA toolkit 13.2 is
  installed. Hence the first round of results is CPU-only.
* Update (same day): after installing the NVIDIA module for the new kernel and
  freeing disk space, the GPU works (driver 595.99.02). The cuGraph-PyG backend
  was then executed and measured — see
  [GRAPH_LEARNING §8.1 and §11](GRAPH_LEARNING.md#81-what-happened-on-a-real-gpu).

## What changed (see git log for details)

Fixed the snapshot JEPA pipeline; added a streaming GH Archive ingester, a
temporal heterogeneous PR graph with leakage-safe splits, PyG / cuGraph-PyG
sampling backends, MLP / HeteroSAGE / HGT node classifiers, a temporal node
JEPA, tabular baselines, multi-seed suites, a throughput benchmark, a
synthetic EV-fleet example, tests and CI. Design and results:
[GRAPH_LEARNING.md](GRAPH_LEARNING.md).
