#!/bin/bash
# Regenerate every talk figure end-to-end. One script per figure.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
PY=.venv/bin/python

echo "== style + schematics (no GPU) =="
$PY -m talk_figs.f7_f8_f9_schematics
$PY -m talk_figs.f9_unet          # LaTeX U-Net (needs pdflatex + pymupdf)

echo "== seismic section figures (no GPU) =="
$PY -m talk_figs.f1_sections
$PY -m talk_figs.f6_sections

echo "== fingerprint / presence (no GPU) =="
$PY -m talk_figs.f5_barcodes
$PY -m talk_figs.f5_sections_bnlogs
$PY -m talk_figs.f8_head_fingerprint
$PY -m talk_figs.f_head_distribution
$PY -m talk_figs.f_head_presence

echo "== GPU figures =="
$PY -m talk_figs.f2_triptych
$PY -m talk_figs.f3_seismic
$PY -m talk_figs.f4_training_movie

echo "== optional =="
$PY -m talk_figs.f10_inversion || echo "(f10 optional)"
$PY -m talk_figs.f11_three_basins || echo "(f11 optional)"

echo "== contact sheet =="
$PY -m talk_figs.contact_sheet

echo "ALL FIGURES DONE -> results/figures/talk/"
