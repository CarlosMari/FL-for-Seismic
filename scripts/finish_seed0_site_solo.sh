#!/usr/bin/env bash
# Replace the 3-seed queue: let the running site finish, run the rest of seed 0, then stop (reboot planned) and score.
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "scripts/[r]un_f3_site_solo.py" >/dev/null; do sleep 30; done
for job in "3 2" "5 0" "5 1" "5 2" "5 3" "5 4"; do
  set -- $job
  [ -f results/runs/seismic_f3_site_solo/fedavg_n$1_site$2_batch/seed_0/results.json ] && continue
  echo "start n=$1 site=$2 seed=0 $(date -Is)"
  .venv/bin/python scripts/run_f3_site_solo.py "$1" "$2" 0 > "logs/f3_site_solo_n$1_s$2_seed0.log" 2>&1 && echo "done n=$1 site=$2 seed=0 $(date -Is)" || echo "FAILED n=$1 site=$2 seed=0"
done
echo "seed 0 finished, GPU free for reboot $(date -Is)"
CUDA_VISIBLE_DEVICES="" .venv/bin/python talk_figs/thesis_ch6_score_site_solo.py > logs/f3_site_solo_score_seed0.log 2>&1 && echo "scored $(date -Is)" || echo "scoring FAILED"
