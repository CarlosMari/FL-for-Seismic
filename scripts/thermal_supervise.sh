#!/usr/bin/env bash
# Run a command under the same conservative GPU thermal policy as the restart.
set -euo pipefail
logdir=${1:?pass a new log directory}; shift
[[ $# -gt 0 && -d "$logdir" && ! -e "$logdir/train.log" ]] || exit 1
echo "$$" >"$logdir/supervisor.pid"
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "${gpu_pids//[[:space:]]/}" ]] || { echo "GPU is in use: $gpu_pids" >&2; exit 1; }
temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
[[ "$temp" =~ ^[0-9]+$ ]] && (( temp < 93 )) || { echo "GPU unavailable or too hot" >&2; exit 1; }

setsid "$@" >"$logdir/train.log" 2>&1 &
worker=$!
echo "$worker" >"$logdir/worker.pid"
stop_worker() {
  local pgid
  pgid=$(ps -o pgid= -p "$worker" 2>/dev/null | tr -d '[:space:]' || true)
  if [[ "$pgid" == "$worker" ]]; then kill -TERM -- "-$worker" 2>/dev/null || true
  else kill -TERM "$worker" 2>/dev/null || true; fi
}
cleanup() {
  if kill -0 "$worker" 2>/dev/null; then
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
    echo "Temperature unavailable; stopping worker" >&2
    stop_worker; wait "$worker" || true
    echo 76 >"$logdir/exit_code"; exit 76
  fi
  if (( temp >= 93 )); then hot=$((hot + 1)); else hot=0; fi
  if (( temp >= 95 || hot >= 3 )); then
    echo "Thermal safeguard: ${temp}C at $(date -Is)" >&2
    stop_worker; wait "$worker" || true
    echo 75 >"$logdir/exit_code"; exit 75
  fi
  sleep 15
done
set +e
wait "$worker"
rc=$?
set -e
echo "$rc" >"$logdir/exit_code"
exit "$rc"
