#!/usr/bin/env bash
# FedHCA2-style vs FedAvg-body (fedpersurvey) vs solo: F3, Parihaka, Penobscot,
# one client each. Finished seeds are skipped, so this is safe to re-run.
# Usage: bash scripts/run_multisurvey3_k3.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for arm in solo_a solo_b solo_c fedpersurvey fedhca2; do
  echo "Starting $arm: $(date -Is)"
  .venv/bin/python -m fedseismic.cli sweep \
    --config "configs/sweeps/multisurvey3_k3_${arm}.json" --device cuda --skip-attack
done
.venv/bin/python scripts/compare_multisurvey3_k3.py --device cuda
echo "Completed: $(date -Is)"
