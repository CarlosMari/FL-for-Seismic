#!/usr/bin/env bash
# Wait for the early-heads chain, then run site-solo seeds 1 and 42 (finished sites skipped).
cd "$(dirname "$0")/.."
while pgrep -f "[c]hain_f3_early_heads" >/dev/null; do sleep 120; done
exec scripts/queue_f3_site_solo_after_reboot.sh
