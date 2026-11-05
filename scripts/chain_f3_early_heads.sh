#!/usr/bin/env bash
# F3 grid, first 5 rounds only, segmentation-head updates logged every round; seed 0 first.
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in 0 1 42; do
  echo "start seed $s $(date -Is)"
  .venv/bin/python -m fedseismic.cli sweep --config configs/sweeps/seismic_f3_early_heads_s$s.json --device cuda --skip-attack > logs/seismic_f3_early_heads_s$s.log 2>&1 && echo "done seed $s $(date -Is)" || echo "FAILED seed $s $(date -Is)"
done
