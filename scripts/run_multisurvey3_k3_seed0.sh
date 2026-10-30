#!/usr/bin/env bash
# Seed-0-only pilot of the prepared three-survey comparison.
# Optionally wait for the PID of the preceding parity analysis to exit.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ $# -gt 0 ]]; then
  while kill -0 "$1" 2>/dev/null; do
    # An orphaned preceding worker may briefly remain as a zombie.
    state=$(ps -o stat= -p "$1" 2>/dev/null || true)
    [[ "$state" == Z* ]] && break
    sleep 60
  done
fi

# Completion markers are insufficient: require both parity seeds and their
# aggregate and per-class rescoring rows before starting the next experiment.
.venv/bin/python - <<'PY'
import csv
from pathlib import Path

arms = {"parity_fl": "fedpersurvey_noniid_a0.1_lNA_dna_n2",
        "parity_f3_solo": "fedpersurvey_noniid_a0.1_lNA_dna_n1",
        "parity_pari_solo": "fedpersurvey_noniid_a0.1_lNA_dna_n1"}
expected = {str(Path("results/runs") / arm / cell / f"seed_{seed}")
            for arm, cell in arms.items() for seed in (0, 1)}
for run in expected:
    if not (Path(run) / "results.json").is_file():
        raise SystemExit(f"parity run not finished: {run}")
for table in ("multisurvey_utility.csv", "multisurvey_per_class.csv"):
    path = Path("results/tables") / table
    with path.open(newline="") as handle:
        actual = {row["run"] for row in csv.DictReader(handle)}
    if missing := expected - actual:
        raise SystemExit(f"{path} lacks parity scores for {sorted(missing)}")
print("Both parity seeds and their aggregate/per-class scores are present", flush=True)
PY

# Avoid colliding with a separately launched training or rescore process.
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "${gpu_pids//[[:space:]]/}" ]]; then
  echo "GPU already in use by: $gpu_pids; pilot not started" >&2
  exit 1
fi

# Do not silently train on uncommitted modifications to the prepared method.
tracked=(fedseismic/config.py fedseismic/experiment.py fedseismic/federated/server.py
         fedseismic/federated/fedhca2.py fedseismic/federated/survey_heads.py
         fedseismic/privacy/multisurvey_utility.py
         configs/seismic_multisurvey3_k3.json scripts/compare_multisurvey3_k3.py
         configs/sweeps/multisurvey3_k3_fedhca2.json
         configs/sweeps/multisurvey3_k3_fedpersurvey.json
         configs/sweeps/multisurvey3_k3_solo_a.json
         configs/sweeps/multisurvey3_k3_solo_b.json
         configs/sweeps/multisurvey3_k3_solo_c.json)
git ls-files --error-unmatch "${tracked[@]}" >/dev/null
git diff --quiet HEAD -- "${tracked[@]}"

config_dir=$(mktemp -d /tmp/opencode/multisurvey3_k3_s0.XXXXXX)
for arm in fedhca2 fedpersurvey solo_a solo_b solo_c; do
  if [[ "$arm" == fedhca2 && -n "${FEDHCA2_OUTPUT_ROOT:-}" ]]; then
    matches=$(find "results/runs/$FEDHCA2_OUTPUT_ROOT" -path '*/seed_0/results.json' -type f 2>/dev/null | wc -l)
    [[ "$matches" -eq 1 ]] || { echo "FedHCA2 restart is incomplete" >&2; exit 1; }
    echo "Using completed FedHCA2 restart: $FEDHCA2_OUTPUT_ROOT"
    continue
  fi
  root="results/runs/multisurvey3_k3_${arm}"
  if [[ -d "$root" ]] && [[ -z "$(find "$root" -path '*/seed_0/results.json' -type f -print -quit)" ]]; then
    echo "Refusing to append to incomplete existing run root $root" >&2
    exit 1
  fi
  original="configs/sweeps/multisurvey3_k3_${arm}.json"
  config="$config_dir/${arm}.json"
  .venv/bin/python - "$original" "$config" <<'PY'
import json
import sys
from pathlib import Path

spec = json.loads(Path(sys.argv[1]).read_text())
spec["seeds"] = [0]
Path(sys.argv[2]).write_text(json.dumps(spec, indent=2) + "\n")
PY
  echo "Starting $arm, seed 0: $(date -Is)"
  .venv/bin/python -m fedseismic.cli sweep --config "$config" --device cuda --skip-attack
done
echo "Seed-0 training complete; rescoring: $(date -Is)"
if [[ -n "${FEDHCA2_OUTPUT_ROOT:-}" ]]; then
  .venv/bin/python scripts/compare_multisurvey3_k3.py --device cuda \
    --fedhca2-root "$FEDHCA2_OUTPUT_ROOT" --out-dir results/tables/multisurvey3_k3_restart_s0
else
  .venv/bin/python scripts/compare_multisurvey3_k3.py --device cuda
fi
echo "Seed-0 comparison complete: $(date -Is)"
echo "NOTE: miou_all_class uses the survey's own class block; miou_union_all_class includes the full union."
