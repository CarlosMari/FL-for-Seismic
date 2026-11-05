#!/usr/bin/env bash
# Resume the paused F3 site-solo job (SIGSTOP'd) once the matched Parihaka solo finishes or dies.
cd "$(dirname "$0")/.."
until ls results/runs/multisurvey_solo_k6_b_rot_buf0/*/seed_0/results.json >/dev/null 2>&1 || ! kill -0 416022 2>/dev/null; do sleep 60; done
pkill -CONT -f "scripts/[r]un_f3_site_solo.py"
echo "resumed site-solo $(date -Is)"
