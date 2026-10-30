#!/usr/bin/env bash
# Called only after the new FedHCA2 control has completed successfully.
set -euo pipefail
cd "$(dirname "$0")/.."
export FEDHCA2_OUTPUT_ROOT=multisurvey3_k3_fedhca2_restart_s0
export PILOT_TABLE_DIR=results/tables/multisurvey3_k3_restart_s0
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/run_multisurvey3_k3_seed0.sh
bash scripts/run_parity_fixed_after_pilot.sh 0
