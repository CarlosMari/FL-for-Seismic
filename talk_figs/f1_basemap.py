"""F1 - survey basemap split into operator blocks + per-operator facies mix.

TOP (map) view of the F3 train survey: faint grayscale backdrop (mean over
depth of the label cube), each operator's contiguous crossline block filled in
its own SITE color, and - on the labeled variant - one horizontal stacked bar
per operator with its facies mix (pi over train pixels, from the run's
labels/pi.json) plus a formation-name legend.

Unlabeled variant: blocks only (letters A/B/C), no bars, no names. The Zechstein
(class 5) map footprint is drawn in dark charcoal so the operator that lacks it
is visually clear; zero-footprint blocks additionally get white hatching and a
heavy navy border.

Reuses repo code: partition_noniid / split_client_local_test (splits) and
privacy.log.load_labels (true per-site histograms). Nothing is reimplemented.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import style
from fedseismic.data.partition import partition_noniid, split_client_local_test
from fedseismic.privacy.log import load_labels

LABELS = ROOT / "datasets/seismic/data/train/train_labels.npy"
RUN_N3 = ROOT / "results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0"
RUN_N5 = ROOT / "results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n5_batchnorm/seed_0"

SEED = 0
LOCAL_TEST_RATIO = 0.2
NUM_CLASSES = 6

# Site colors for the operator blocks. Deliberately NOT the FACIES_COLORS hues
# (Okabe-Ito: 17/33/53/158/195/202/330 deg) so site identity can never be
# confused with facies identity. Hues sit in the two free windows (~255-290
# violet/purple, ~85-120 yellow-green) plus a desaturated taupe; no dark
# neutrals, so the dark Zechstein footprint stays visible on every block.
SITE_COLORS = ["#4A2D8E", "#4F7A24", "#8E2E9E", "#4CA04C", "#A89B8C"]
LETTERS = "ABCDE"

C5_RGB = (0.07, 0.07, 0.07, 0.95)  # Zechstein footprint in the unlabeled variant


def _faint_gray(backdrop):
    """Scale a backdrop array to a faint 0.66-0.98 grayscale range."""
    b = np.asarray(backdrop, dtype=np.float64)
    b = (b - b.min()) / (np.ptp(b) + 1e-8)
    return 0.66 + 0.32 * b


def _split(N, labels_cube, run_dir):
    """Everything the figure needs for one N-operator split."""
    parts = partition_noniid(labels_cube.shape[1], N)
    train_parts, test_parts = split_client_local_test(
        parts, LOCAL_TEST_RATIO, rng=None)
    pi_by_client, rare = load_labels(run_dir)
    stats = []
    for i, (blk, trp, tep) in enumerate(zip(parts, train_parts, test_parts)):
        block = labels_cube[:, blk, :]
        c5 = block == 5
        stats.append({
            "client": i,
            "letter": LETTERS[i],
            "block": [int(blk[0]), int(blk[-1])],
            "n_xlines": len(blk),
            "train": [int(trp[0]), int(trp[-1])],
            "holdout_tail": [int(tep[0]), int(tep[-1])],
            "pi_train": [round(float(v), 6) for v in pi_by_client[i]],
            "pi_full_block": [
                round(float((block == c).mean()), 6) for c in range(NUM_CLASSES)],
            "c5_pixels_full": float(c5.mean()),
            "c5_pixels_train": float((labels_cube[:, trp, :] == 5).mean()),
            "c5_map_frac": float(c5.any(axis=2).mean()),
            "lacks_c5_full": bool(not c5.any()),
            "lacks_c5_train": bool(pi_by_client[i][5] == 0.0),
        })
    return parts, train_parts, test_parts, stats, rare


def _operator_name(stat):
    return f"Operator {stat['letter']}"


def _draw_map_tall(ax, faint, stats, tails, show_tail_hatch=True):
    """Map view, x=inline (401), y=crossline (701): true index aspect.

    Blocks are horizontal bands (top->bottom follows the operator order);
    labels, optional local-holdout hatch on each block's test tail."""
    ax.imshow(faint.T, cmap="gray", vmin=0, vmax=1, aspect="equal",
              interpolation="nearest")
    for stat in stats:
        lo, hi = stat["block"]
        color = SITE_COLORS[stat["client"]]
        ax.axhspan(lo - 0.5, hi + 0.5, color=color, alpha=0.55, lw=0)
        ax.axhline(hi + 0.5, color="white", lw=2.5)
        ax.axhline(lo - 0.5, color="white", lw=2.5)
    if show_tail_hatch:
        for tail in tails:
            ax.axhspan(tail[0] - 0.5, tail[1] + 0.5, facecolor="none",
                       hatch="////", edgecolor="white", linewidth=0)
    for stat in stats:
        lo, hi = stat["block"]
        ax.text(200.5, 0.5 * (lo + hi), _operator_name(stat), ha="center",
                va="center", fontsize=18, weight="bold", color=style.NAVY,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none",
                          alpha=0.85))
    ax.set_xlim(-0.5, faint.shape[0] - 0.5)
    ax.set_ylim(faint.shape[1] - 0.5, -0.5)  # crossline 0 at top: A/B/C down
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_bars(ax, stats, pi_key="pi_train"):
    """One horizontal stacked facies bar per operator (top->bottom A, B, C...)."""
    cmap, _ = style.facies_cmap(NUM_CLASSES)
    n = len(stats)
    for i, stat in enumerate(stats):
        pi = np.asarray(stat[pi_key], dtype=float)
        ax.text(0.0, i - 0.34, _operator_name(stat), ha="left", va="bottom",
                fontsize=16, weight="bold", color=style.NAVY)
        left = 0.0
        for c in range(NUM_CLASSES):
            if pi[c] <= 0:
                continue
            ax.barh(i, pi[c], left=left, height=0.42,
                    color=style.FACIES_COLORS[c], edgecolor="white", lw=1.0)
            left += pi[c]
    ax.set_xlim(0, 1)
    ax.set_ylim(n - 0.45, -0.62)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_legend(ax):
    handles = [mpatches.Patch(color=style.FACIES_COLORS[c], label=style.F3_NAMES[c])
               for c in range(NUM_CLASSES)]
    ax.legend(handles=handles, loc="upper left", ncol=2, frameon=False,
              fontsize=14, handlelength=1.4, columnspacing=1.2, labelspacing=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_labeled(N, labels_cube, run_dir, stem):
    parts, train_parts, test_parts, stats, rare = _split(N, labels_cube, run_dir)
    faint = _faint_gray(labels_cube.mean(axis=2))

    # --- geometry: tall true-aspect map left, bars right -> ~1.1:1 figure.
    # save() adds 0.3in of tight-bbox padding; solve (W+0.3)/(H+0.3)=1.1.
    h = 8.60
    w = 1.1 * (h + 0.3) - 0.3
    fig = plt.figure(figsize=(w, h))
    fig.add_artist(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                 fc="white", ec="none", zorder=-10))

    map_h = h - 0.16
    map_w = map_h * labels_cube.shape[0] / labels_cube.shape[1]
    ax_map = fig.add_axes([0.08 / w, 0.08 / h, map_w / w, map_h / h])
    _draw_map_tall(ax_map, faint, stats, [s["holdout_tail"] for s in stats])

    bars_x = 0.08 + map_w + 0.22
    bars_w = w - bars_x - 0.08
    legend_h = 1.55
    ax_bars = fig.add_axes([bars_x / w, (0.08 + legend_h + 0.15) / h, bars_w / w,
                            (map_h - legend_h - 0.15) / h])
    _draw_bars(ax_bars, stats, pi_key="pi_train")
    ax_leg = fig.add_axes([bars_x / w, 0.08 / h, bars_w / w, legend_h / h])
    _draw_legend(ax_leg)

    style.save(fig, stem)
    return stats, rare


def make_unlabeled(N, labels_cube, stem):
    parts, train_parts, test_parts, stats, rare = _split(N, labels_cube, RUN_N3 if N == 3 else RUN_N5)
    faint = _faint_gray(labels_cube.mean(axis=2))
    c5_mask = (labels_cube == 5).any(axis=2)  # Zechstein map footprint (inline, crossline)

    # --- geometry: wide map (x=crossline stripes, y=inline) filling a 1.4:1 box.
    h = 7.40
    w = 1.4 * (h + 0.3) - 0.3
    fig = plt.figure(figsize=(w, h))
    fig.add_artist(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                 fc="white", ec="none", zorder=-10))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])

    ax.imshow(faint, cmap="gray", vmin=0, vmax=1, aspect="auto",
              interpolation="nearest")
    for stat in stats:
        lo, hi = stat["block"]
        color = SITE_COLORS[stat["client"]]
        ax.axvspan(lo - 0.5, hi + 0.5, color=color, alpha=0.55, lw=0)
        ax.axvline(hi + 0.5, color="white", lw=2.5)
        ax.axvline(lo - 0.5, color="white", lw=2.5)

    # Zechstein (class 5) map footprint: present in dark charcoal, absent blank.
    rgba = np.zeros(c5_mask.shape + (4,), dtype=float)
    rgba[c5_mask] = C5_RGB
    ax.imshow(rgba, aspect="auto", interpolation="nearest")

    # Operators with ZERO Zechstein anywhere in their block: hatch + heavy border
    # so the "who has no Zechstein?" answer is unambiguous from the blocks alone.
    for stat in stats:
        if not stat["lacks_c5_full"]:
            continue
        lo, hi = stat["block"]
        ax.add_patch(mpatches.Rectangle(
            (lo - 0.5, -0.5), hi - lo + 1, faint.shape[0], facecolor="none",
            hatch="////", edgecolor="white", linewidth=0))
        ax.add_patch(mpatches.Rectangle(
            (lo - 0.5, -0.5), hi - lo + 1, faint.shape[0], facecolor="none",
            edgecolor=style.NAVY, linewidth=3.0))

    for stat in stats:
        lo, hi = stat["block"]
        ax.text(0.5 * (lo + hi), 0.5 * (faint.shape[0] - 1), stat["letter"],
                ha="center", va="center", fontsize=34, weight="bold",
                color=style.NAVY,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none",
                          alpha=0.85))
    ax.set_xlim(-0.5, faint.shape[1] - 0.5)
    ax.set_ylim(faint.shape[0] - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    style.save(fig, stem)
    return stats, rare


def main():
    labels_cube = np.load(LABELS)
    facts = {"labels_cube": str(LABELS.relative_to(ROOT)),
             "shape": list(labels_cube.shape), "runs": {}}

    for N in (3, 5):
        run_dir = RUN_N3 if N == 3 else RUN_N5
        stem = f"f1_basemap_n{N}"
        stats, rare = make_labeled(N, labels_cube, run_dir, stem)
        u_stats, _ = make_unlabeled(N, labels_cube, f"f1_basemap_unlabeled_n{N}")
        facts["runs"][f"n{N}"] = {
            "run": str(run_dir.relative_to(ROOT)),
            "stats": stats,
            "n_lacking_c5_full": sum(s["lacks_c5_full"] for s in stats),
            "n_lacking_c5_train": sum(s["lacks_c5_train"] for s in stats),
            "rare_classes": list(rare),
        }
        print(f"  N={N}: "
              f"{sum(s['lacks_c5_full'] for s in stats)}/{N} operators have zero "
              f"class-5 pixels in their block; "
              f"{sum(s['lacks_c5_train'] for s in stats)}/{N} in their train split")

    out = HERE / "_facts_f1.json"
    out.write_text(json.dumps(facts, indent=1), encoding="utf-8")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
