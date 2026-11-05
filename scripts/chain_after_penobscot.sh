#!/usr/bin/env bash
# After the Penobscot solo finishes: GPU rescores (old multi-survey audit, three-survey k3), then the F3 site-alone queue.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" DEVICE=cuda
until ls results/runs/multisurvey3_k3_solo_c/*/seed_0/results.json >/dev/null 2>&1; do
  pgrep -f "solo_c_s0.json" >/dev/null || { echo "Penobscot run gone without results; stopping $(date -Is)"; exit 1; }
  sleep 60
done
echo "penobscot done $(date -Is)"
.venv/bin/python talk_figs/thesis_ch6_rescore_k3.py > logs/rescore_k3.log 2>&1 && echo "k3 rescore done $(date -Is)" || echo "k3 rescore FAILED"
scripts/queue_f3_site_solo.sh > logs/f3_site_solo.log 2>&1
