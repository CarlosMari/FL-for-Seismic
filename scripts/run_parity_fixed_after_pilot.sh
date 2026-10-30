#!/usr/bin/env bash
# Run the original 350-section comparison with the fixed implementation.
# Wait for the complete three-survey seed-0 pilot, not just its first arm.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

parent=${1:?pass the pilot process ID, or 0 after the pilot completes}
if [[ "$parent" != 0 ]]; then
  while kill -0 "$parent" 2>/dev/null; do
    state=$(ps -o stat= -p "$parent" 2>/dev/null || true)
    [[ "$state" == Z* ]] && break
    sleep 60
  done
fi

for arm in fedhca2 fedpersurvey solo_a solo_b solo_c; do
  if [[ "$arm" == fedhca2 && -n "${FEDHCA2_OUTPUT_ROOT:-}" ]]; then
    root="results/runs/$FEDHCA2_OUTPUT_ROOT"
  else
    root="results/runs/multisurvey3_k3_${arm}"
  fi
  if [[ $(find "$root" -path '*/seed_0/results.json' -type f 2>/dev/null | wc -l) -ne 1 ]]; then
    echo "Three-survey seed-0 arm $arm is incomplete; parity not started" >&2
    exit 1
  fi
done
table_dir=${PILOT_TABLE_DIR:-results/tables}
for table in multisurvey3_k3_runs.csv multisurvey3_k3_per_class.csv; do
  [[ -s "$table_dir/$table" ]] || { echo "Missing $table; parity not started" >&2; exit 1; }
done
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "${gpu_pids//[[:space:]]/}" ]]; then
  echo "GPU already in use by $gpu_pids; parity not started" >&2
  exit 1
fi
tracked=(fedseismic/config.py fedseismic/experiment.py fedseismic/federated/server.py
         fedseismic/federated/client.py fedseismic/federated/survey_heads.py
         fedseismic/data/partition.py fedseismic/data/seismic.py fedseismic/models/unet.py
         fedseismic/privacy/multisurvey_utility.py
         configs/parity_fl.json configs/parity_f3_solo.json configs/parity_pari_solo.json
         configs/sweeps/parity_fl.json configs/sweeps/parity_f3_solo.json
         configs/sweeps/parity_pari_solo.json
         scripts/score_parity_fixed.py)
git ls-files --error-unmatch "${tracked[@]}" >/dev/null
git diff --quiet HEAD -- "${tracked[@]}"

config_dir=$(mktemp -d /tmp/opencode/parity_fixed.XXXXXX)
for seed in 0 1; do
  for arm in parity_fl parity_f3_solo parity_pari_solo; do
    config="$config_dir/${arm}_seed${seed}.json"
    .venv/bin/python - "configs/sweeps/${arm}.json" "$config" "$seed" <<'PY'
import json
import sys
from pathlib import Path

spec = json.loads(Path(sys.argv[1]).read_text())
spec["seeds"] = [int(sys.argv[3])]
spec["output_root"] += "_fixed"
Path(sys.argv[2]).write_text(json.dumps(spec, indent=2) + "\n")
PY
    echo "Starting corrected $arm, seed $seed: $(date -Is)"
    .venv/bin/python -m fedseismic.cli sweep --config "$config" --device cuda --skip-attack
  done
done
.venv/bin/python scripts/score_parity_fixed.py --device cuda
echo "Corrected two-seed comparison complete: $(date -Is)"
