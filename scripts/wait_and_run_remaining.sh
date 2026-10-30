#!/usr/bin/env bash
# Detached, fail-closed handoff after the monitored seed-0 control.
set -euo pipefail
cd "$(dirname "$0")/.."
control_pid=${1:?pass the control supervisor PID}
control_logs=${2:?pass the control log directory}
while kill -0 "$control_pid" 2>/dev/null; do
  state=$(ps -o stat= -p "$control_pid" 2>/dev/null || true)
  [[ "$state" == Z* ]] && break
  sleep 30
done
run=results/runs/multisurvey3_k3_fedhca2_restart_s0/fedhca2_noniid_a0.1_lNA_dna_n3/seed_0
if [[ ! -f "$control_logs/exit_code" || $(cat "$control_logs/exit_code") != 0 ||
      ! -f "$run/results.json" || ! -f "$run/global_final.pt" ]]; then
  echo "Control did not complete successfully; no follow-up jobs started" >&2
  exit 1
fi

# Let the device cool between long jobs. A failure to cool is a hard stop.
cooled=0
for _ in $(seq 1 60); do
  temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
  if [[ "$temp" =~ ^[0-9]+$ ]] && (( temp <= 85 )); then
    cooled=$((cooled + 1))
    (( cooled >= 2 )) && break
  else
    cooled=0
  fi
  sleep 60
done
if (( cooled < 2 )); then
  echo "GPU did not cool sufficiently; no follow-up jobs started" >&2
  exit 1
fi
mkdir -p results/logs
logdir=$(mktemp -d "$PWD/results/logs/fedhca2_followup_s0_XXXXXX")
echo "$logdir" >"$control_logs/followup_log_dir"
echo "Starting remaining seed-0 pilot and corrected parity: $(date -Is)"
exec bash scripts/thermal_supervise.sh "$logdir" bash scripts/run_remaining_pilot_and_parity.sh \
  >"$logdir/supervisor.log" 2>&1
