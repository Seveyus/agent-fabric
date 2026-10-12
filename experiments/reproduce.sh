#!/usr/bin/env bash
# Reproduce every result in experiments/results/ (CPU, ~3-4 h end to end on a
# 4-core laptop; the GH Archive download is ~60 GB streamed at ~10 MB/s).
# Usage: bash experiments/reproduce.sh [step ...]   (default: all steps)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
steps=${*:-"ingest synthetic synthetic_jepa ev gh jepa bench"}

for step in $steps; do
  case "$step" in
    ingest)   # 1. GH Archive: warm-up week in full, then collaborative repos only
      $PY -m core.dataset.gharchive_ingest ingest --start 2025-08-25 --end 2025-08-31 --out data/gharchive/warmup_raw
      $PY -m core.dataset.gharchive_ingest select-repos --events data/gharchive/warmup_raw --out data/gharchive/repos.txt
      $PY -m core.dataset.gharchive_ingest filter --events data/gharchive/warmup_raw --repos data/gharchive/repos.txt --out data/gharchive/events
      $PY -m core.dataset.gharchive_ingest ingest --start 2025-09-01 --end 2025-09-15 --out data/gharchive/events \
          --repo-filter data/gharchive/repos.txt ;;
    synthetic) # 2. synthetic PR graph (planted mechanism)
      $PY -m training.suite --suite main --config configs/synthetic_pr_merge.toml --seeds 0 1 2 ;;
    synthetic_jepa)
      $PY -m training.suite --suite jepa --config configs/synthetic_pr_merge.toml --seeds 0 1 2 ;;
    ev)       # 3. synthetic EV fleet
      $PY -m training.suite --suite main --config configs/ev_fleet.toml --seeds 0 1 2 ;;
    gh)       # 4. real GH Archive graph
      $PY -m training.suite --suite main --config configs/gh_pr_merge.toml --seeds 0 1 2 ;;
    jepa)
      $PY -m training.suite --suite jepa --config configs/gh_pr_merge.toml --seeds 0 1 2 ;;
    bench)    # 5. throughput (run alone on the machine)
      $PY -m benchmarks.sampling_throughput --backend pyg --scales 2000 20000 60000 \
          --task data/gharchive/task_pr_merge_48h.pt --batches 20 --tag cpu ;;
    *) echo "unknown step $step" >&2; exit 1 ;;
  esac
done
