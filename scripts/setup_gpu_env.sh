#!/usr/bin/env bash
# Create .venv-gpu (CUDA PyTorch + PyG + pyg-lib + RAPIDS cuGraph-PyG), check the
# GPU is usable, run the cuGraph tests and the GPU throughput benchmark.
# The CPU environment (.venv, requirements/graph-cpu.txt) is left untouched.
#
#   bash scripts/setup_gpu_env.sh            # install + verify + benchmark
#   bash scripts/setup_gpu_env.sh --no-bench # install + verify only
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=.venv-gpu
NEED_GB=11

if ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi fails: the NVIDIA kernel module is not loaded for kernel $(uname -r)." >&2
  echo "Install it:  sudo apt install linux-modules-nvidia-595-open-generic-hwe-24.04 && sudo reboot" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv

if [ ! -x "$VENV/bin/python" ]; then
  free_gb=$(df --output=avail -BG . | tail -1 | tr -dc '0-9')
  if [ "$free_gb" -lt "$NEED_GB" ]; then
    echo "Need ~${NEED_GB} GB free on this filesystem, only ${free_gb} GB available." >&2
    exit 1
  fi
  python3 -m venv "$VENV"
  if command -v uv >/dev/null 2>&1; then
    # uv unpacks into its cache and hard-links into the venv: peak disk use is
    # about the installed size (~10 GB). pip keeps every downloaded wheel
    # (~6 GB compressed) until the end, so its peak is ~16 GB.
    UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 uv pip install --python "$VENV/bin/python" \
        --link-mode hardlink --index-strategy unsafe-best-match -r requirements/graph-cugraph-cu12.txt
  else
    echo "uv not found: falling back to pip (needs ~16 GB free during the install)" >&2
    "$VENV/bin/pip" install --no-cache-dir --upgrade pip
    "$VENV/bin/pip" install --no-cache-dir -r requirements/graph-cugraph-cu12.txt
  fi
fi

"$VENV/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "torch does not see the GPU"
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(), "| CC", torch.cuda.get_device_capability())
from training.backends import cugraph_available
ok, reason = cugraph_available()
print("cuGraph backend available:", ok, "-", reason)
assert ok, reason
PY

# 2 tests must pass (seed alignment, 1-hop temporal correctness); the 2-hop
# temporal test is a strict xfail documenting a cugraph-pyg 26.10 issue.
PYTHONPATH=. "$VENV/bin/python" -m pytest tests/ml -q -m cugraph -rxs
PYTHONPATH=. "$VENV/bin/python" -m pytest tests/ml -q -m "not slow"
PYTHONPATH=. "$VENV/bin/python" scripts/cugraph_temporal_check.py --seeds 60

if [ "${1:-}" != "--no-bench" ]; then
  # PyG sampler + model on the GPU, then cuGraph sampler + model on the GPU, same graphs.
  PYTHONPATH=. "$VENV/bin/python" -m benchmarks.sampling_throughput --backend pyg --device cuda \
      --scales 2000 20000 --task data/gharchive/task_pr_merge_48h.pt --batches 20 --tag pyg_cuda
  PYTHONPATH=. "$VENV/bin/python" -m benchmarks.sampling_throughput --backend cugraph --device cuda \
      --scales 2000 20000 --task data/gharchive/task_pr_merge_48h.pt --batches 20 --tag cugraph_cuda
fi
