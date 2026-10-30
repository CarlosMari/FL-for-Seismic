#!/usr/bin/env bash
# Single-seed restart of the planned hyperweight control, in a NEW run root.
# Usage: bash scripts/run_fedhca2_restart_s0.sh <new-log-directory>
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

logdir=${1:?pass a new, dedicated log directory}
root=results/runs/multisurvey3_k3_fedhca2_restart_s0
config=configs/sweeps/multisurvey3_k3_fedhca2_restart_s0.json
if [[ -e "$root" ]]; then
  echo "Refusing to overwrite or append to $root" >&2
  exit 1
fi
if [[ ! -d "$logdir" || -e "$logdir/train.log" ]]; then
  echo "Log directory must exist and contain no train.log: $logdir" >&2
  exit 1
fi
echo "$$" >"$logdir/supervisor.pid"
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "${gpu_pids//[[:space:]]/}" ]]; then
  echo "GPU already in use by $gpu_pids" >&2
  exit 1
fi
initial_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
if [[ ! "$initial_temp" =~ ^[0-9]+$ ]] || (( initial_temp >= 93 )); then
  echo "GPU temperature unavailable or at/above 93C; refusing to start" >&2
  exit 1
fi

echo "Starting seed 0 at $(date -Is); logs: $logdir"
setsid .venv/bin/python -m fedseismic.cli sweep --config "$config" --device cuda --skip-attack \
  >"$logdir/train.log" 2>&1 &
worker=$!
echo "$worker" >"$logdir/worker.pid"
stop_worker() {
  local pgid
  pgid=$(ps -o pgid= -p "$worker" 2>/dev/null | tr -d '[:space:]' || true)
  if [[ "$pgid" == "$worker" ]]; then
    kill -TERM -- "-$worker" 2>/dev/null || true
  else
    kill -TERM "$worker" 2>/dev/null || true
  fi
}
cleanup() {
  if kill -0 "$worker" 2>/dev/null; then
    echo "Supervisor exiting while worker is live; stopping its process group" >&2
    stop_worker
    wait "$worker" 2>/dev/null || true
  fi
}
trap cleanup EXIT
hot=0
while kill -0 "$worker" 2>/dev/null; do
  sample=$(nvidia-smi --query-gpu=timestamp,temperature.gpu,fan.speed,utilization.gpu,memory.used,power.draw,clocks.current.sm,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader) || sample=""
  echo "$sample" >>"$logdir/gpu.csv"
  temp=$(printf '%s\n' "$sample" | cut -d, -f2 | tr -d '[:space:]')
  if [[ ! "$temp" =~ ^[0-9]+$ ]]; then
    echo "Cannot read GPU temperature; stopping the worker" >&2
    stop_worker
    wait "$worker" || true
    echo 76 >"$logdir/exit_code"
    exit 76
  fi
  if (( temp >= 93 )); then hot=$((hot + 1)); else hot=0; fi
  if (( temp >= 95 || hot >= 3 )); then
    echo "Thermal safeguard: ${temp}C; stopping worker at $(date -Is)" >&2
    stop_worker
    wait "$worker" || true
    echo 75 >"$logdir/exit_code"
    exit 75
  fi
  sleep 15
done
set +e
wait "$worker"
rc=$?
set -e
echo "$rc" >"$logdir/exit_code"
if (( rc != 0 )); then
  echo "Training failed with exit code $rc at $(date -Is)" >&2
  exit "$rc"
fi
run="$root/fedhca2_noniid_a0.1_lNA_dna_n3/seed_0"
if [[ ! -f "$run/results.json" || ! -f "$run/global_final.pt" ]]; then
  echo "Training returned without the expected final artifacts" >&2
  exit 1
fi
echo "Seed-0 control completed at $(date -Is)"
