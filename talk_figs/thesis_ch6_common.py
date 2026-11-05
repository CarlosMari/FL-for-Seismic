"""Shared helpers for the Chapter 6 thesis figures (figures/ch6/).

Imports the facies palette and names from talk_figs/style.py and replaces its
slide rcParams with the thesis style: 6.5 in wide, Computer Modern, 9 pt body,
8 pt minimum, no titles, exact figure size (no tight bbox).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "talk_figs"))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import style

OUT = ROOT / "figures/ch6"
DATA = OUT / "data"
W = 6.5
SITE_COLORS = ["#2F2F2F", "#8C1D40", "#6A3D9A", "#6B8E23", "#0F6B78"]
SITE_MARKERS = ["o", "s", "^", "D", "v"]
LETTERS = "ABCDE"
EVAL_THREADS = 10

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "hatch.linewidth": 0.5,
    "pdf.fonttype": 42,
    "savefig.bbox": None,
    "savefig.pad_inches": 0,
    "savefig.dpi": 300,
    "figure.dpi": 100,
})


def inch_axes(fig, x, y, w, h):
    """Axes from an inch rectangle, x and y measured from the top-left."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, (fh - y - h) / fh, w / fw, h / fh])


def plain(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
    return ax


def panel_label(ax, text, inside=True):
    if inside:
        ax.text(0.015, 0.975, text, transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=20)
    else:
        ax.text(0, 1.02, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=9)


def corner_note(ax, text, loc="lower right"):
    x = 0.985 if "right" in loc else 0.015
    y = 0.03 if "lower" in loc else 0.97
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8,
            ha="right" if "right" in loc else "left",
            va="bottom" if "lower" in loc else "top",
            bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=20)


def seismic(ax, section):
    # The loaders min-max normalize to [0, 1], so clip about the data, not about 0.
    vmin, vmax = np.percentile(section, [2, 98])
    ax.imshow(section, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
    return plain(ax)


def facies_over(ax, section, classes, n=6, alpha=0.55):
    """Seismic in grey with facies at 0.55 alpha; 255 (non-facies) left clear."""
    seismic(ax, section)
    cmap, norm = style.facies_cmap(n)
    ax.imshow(np.ma.masked_where(classes == 255, classes), cmap=cmap, norm=norm, alpha=alpha,
              aspect="equal", interpolation="nearest")
    return ax


def error_map(ax, truth, pred, n=6):
    """White where correct, the true-facies colour where wrong."""
    wrong = np.where((pred == truth) | (truth == 255), -1, truth)
    cmap, norm = style.error_cmap(n)
    ax.imshow(wrong, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
    return plain(ax)


def facies_handles(names):
    return [mpatches.Patch(fc=style.FACIES_COLORS[k], ec="none", label=name)
            for k, name in enumerate(names)]


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    plt.close(fig)
    print("wrote", OUT / f"{stem}.pdf", flush=True)


def write_facts(stem, facts):
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{stem}.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
