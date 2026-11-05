#!/usr/bin/env bash
# After the 2026-11-03 reboot: every F3 site trained alone, seed 0 first; finished sites are skipped by run_f3_site_solo.py.
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for seed in 0 1 42; do
  for n in 3 5; do for ((site = 0; site < n; site++)); do
    [ -f results/runs/seismic_f3_site_solo/fedavg_n${n}_site${site}_batch/seed_$seed/results.json ] && continue
    echo "start n=$n site=$site seed=$seed $(date -Is)"
    .venv/bin/python scripts/run_f3_site_solo.py $n $site $seed > logs/f3_site_solo_n${n}_s${site}_seed$seed.log 2>&1 \
      && echo "done n=$n site=$site seed=$seed $(date -Is)" || echo "FAILED n=$n site=$site seed=$seed $(date -Is)"
  done; done
  echo "seed $seed complete $(date -Is)"
done
echo "all site-solo runs finished $(date -Is)"
