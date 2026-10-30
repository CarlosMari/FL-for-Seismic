#!/usr/bin/env bash
# Wait for the current parity sweep and its rescore to exit before using CUDA.
# Usage: bash scripts/queue_fedavg_after_parity.sh <parity-parent-pid>
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

parent_pid=${1:?pass the parity chaser PID}
while kill -0 "$parent_pid" 2>/dev/null; do
  sleep 60
done

config=configs/sweeps/multisurvey_fedavg_rot12.json
run=results/runs/multisurvey_fedavg_rot12/fedavg_noniid_a0.1_lNA_dna_n4/seed_0
echo "Starting 12-round FedAvg: $(date -Is)"
.venv/bin/python -m fedseismic.cli sweep --config "$config" --device cuda --skip-attack
if [[ ! -f "$run/results.json" || ! -f "$run/global_final.pt" ]]; then
  echo "FedAvg returned without the expected result and checkpoint" >&2
  exit 1
fi
echo "FedAvg completed: $(date -Is)"
.venv/bin/python -m talk_figs.f3_seismic --run "$run" --fedbn "" --tag corrected_rot12
echo "Blackout figure completed: $(date -Is)"
