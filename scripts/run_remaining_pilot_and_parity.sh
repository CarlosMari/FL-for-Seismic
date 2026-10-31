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
PARITY_SEEDS=0 bash scripts/run_parity_fixed_after_pilot.sh 0
# Exploratory 17-round lower-hyper-LR arm (dated note 2026-10-31 in findings.md):
# saturation check only, not comparable to 20-round arms.
.venv/bin/python -m fedseismic.cli sweep \
  --config configs/sweeps/multisurvey3_k3_fedhca2_hlr1e4_s0.json \
  --device cuda --skip-attack
