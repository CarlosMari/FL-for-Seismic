#!/usr/bin/env bash
# Wait for the Penobscot solo run, then train every F3 site alone, one at a time (seed 0 first).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until ls results/runs/multisurvey3_k3_solo_c/*/seed_0/results.json >/dev/null 2>&1; do
  pgrep -f "solo_c_s0.json" >/dev/null || { echo "Penobscot run gone without results; stopping"; exit 1; }
  sleep 60
done
for seed in 0 1 42; do for n in 3 5; do for ((site = 0; site < n; site++)); do echo "$n $site $seed"; done; done; done |
  xargs -P 1 -L 1 bash -c 'echo "start n=$0 site=$1 seed=$2 $(date -Is)"; .venv/bin/python scripts/run_f3_site_solo.py "$0" "$1" "$2" > "logs/f3_site_solo_n$0_s$1_seed$2.log" 2>&1 && echo "done n=$0 site=$1 seed=$2 $(date -Is)" || echo "FAILED n=$0 site=$1 seed=$2"'
echo "all site-solo runs finished $(date -Is)"
