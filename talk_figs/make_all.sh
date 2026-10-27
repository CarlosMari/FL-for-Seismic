#!/bin/bash
# Run every talk-figure script end to end, in order. Reproducible.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
PY=.venv/bin/python

echo "== style + schematics (no GPU) =="
$PY -m talk_figs.f7_f8_f9_schematics

echo "== F1 operators as sections (no GPU) =="
$PY -m talk_figs.f1_sections

echo "== F6 who has the Zechstein (no GPU) =="
$PY -m talk_figs.f6_sections

echo "== F5 fingerprint: sections + BN logs (no GPU) =="
$PY -m talk_figs.f5_sections_bnlogs

echo "== F2 triptych (GPU) =="
$PY -m talk_figs.f2_triptych

echo "== F3 blackout + F4 movie (GPU) =="
$PY -m talk_figs.f3_blackout
$PY -m talk_figs.f4_training_movie

echo "== backup variants (PCA fingerprint + rank x-ray) =="
$PY -m talk_figs.f5_fingerprint || echo "  (f5_fingerprint optional, skipped)"
$PY -m talk_figs.f6_xray || echo "  (f6_xray optional, skipped)"

echo "== contact sheet =="
$PY -m talk_figs.contact_sheet

echo "ALL FIGURES DONE -> results/figures/talk/"
