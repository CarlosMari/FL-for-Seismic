#!/usr/bin/env bash
# Reruns of the method grids (p0 = LA, p1 = FedAvg) that keep the received model of every round.
cd "$(dirname "$0")/.."
for d in blood organc cifar; do
  .venv/bin/python -m fedseismic.cli sweep --config configs/sweeps/single_upload_rerun_$d.json --device cuda --skip-attack \
    > logs/single_upload_rerun_$d.log 2>&1 && echo "done $d $(date -Is)" || echo "FAILED $d $(date -Is)"
done
