#!/usr/bin/env bash
# Fixed head aggregation, two-survey protocol: multi-survey run, then the Parihaka solo, then resume the paused F3 site queue.
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for n in multisurvey_fedpersurvey_rot_fixedhead multisurvey_solo_k6_b_rot_fixedhead; do
  echo "start $n $(date -Is)"
  .venv/bin/python -m fedseismic.cli sweep --config configs/sweeps/$n.json --device cuda --skip-attack > logs/$n.log 2>&1 && echo "done $n $(date -Is)" || echo "FAILED $n $(date -Is)"
done
pkill -CONT -f "scripts/[r]un_f3_site_solo.py"
echo "resumed site queue $(date -Is)"
