#!/usr/bin/env bash
# Tier 0/1 campaign (plan.md, 2026-08-03):
#   A) equal-agg FedAvg control, n=5      -> true baseline for the ablation ladder
#   B) GroupNorm + equal agg, n=5         -> isolates the BatchNorm non-IID failure
#   C) GroupNorm + T agg, n=5             -> best agg (on final-round) + GroupNorm
# Sequential: one 3090, ~40 min/run, 15 runs total (~10h).
set -u

cd /home/carlos/projects/FL-for-Seismic
LOGDIR=logs_tier01
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"
SEEDS=(42 123 7 99 2025)
COMMON="--split noniid --num_clients 20 --num_rounds 20 \
--sample_ratio 0.25 --loss recall_slice"

echo "=== Tier 0/1 campaign started: $(date -u) ===" | tee -a "$MASTER"

run () {           # run <tag> <norm> <agg> <extra...>
  local tag=$1 norm=$2 agg=$3; shift 3
  for s in "${SEEDS[@]}"; do
    local log="$LOGDIR/${tag}_s${s}.log"
    if [ -s "$log" ] && grep -q "Training complete" "$log"; then
      echo "SKIP ${tag}_s${s} (already complete)" | tee -a "$MASTER"; continue
    fi
    echo "--> ${tag}_s${s} norm=${norm} agg=${agg} start $(date -u +%H:%M)" | tee -a "$MASTER"
    UNET_NORM="$norm" python3 train_federated.py $COMMON \
        --agg_strategy "$agg" --seed "$s" "$@" >"$log" 2>&1
    if grep -q "Training complete" "$log"; then
      echo "    OK  ${tag}_s${s}  best=$(grep -oP 'Best average mIoU:\s*\K[0-9.]+' "$log" | tail -1)" | tee -a "$MASTER"
    else
      echo "    FAIL ${tag}_s${s} -- see $log" | tee -a "$MASTER"
      tail -5 "$log" | sed 's/^/         /' | tee -a "$MASTER"
    fi
  done
}

# A) true FedAvg control (Tier 0.3)
run A_equal_bn    batch equal

# B) GroupNorm, same agg -> clean one-variable comparison vs A (Tier 1.1)
run B_equal_gn    group equal

# C) GroupNorm + best final-round agg
run C_t_gn        group invfreq_invmiou

echo "=== Tier 0/1 campaign done: $(date -u) ===" | tee -a "$MASTER"
