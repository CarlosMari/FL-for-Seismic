"""Shared style for the sponsor talk figures.

One fixed facies palette (colorblind-safe, stratigraphic order) used in EVERY
figure, plus seismic-display helpers, aspect ratios and fonts. Keep all
figures consistent by importing from here instead of hand-rolling colors.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

# ---------------------------------------------------------------- fonts / style
FONT = "DejaVu Sans"   # Arial-metric sans; swap to "Arial" where installed
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

NAVY = "#003057"
GOLD = "#B3A369"

# ---------------------------------------------------------------- facies palette
# Colorblind-safe (Okabe-Ito base), assigned in STRATIGRAPHIC order (shallow ->
# deep). NOT tab10. One palette shared across every figure. Order matters.
FACIES_COLORS = [
    "#0072B2",  # 0 shallow  (blue)
    "#56B4E9",  # 1          (sky blue)
    "#009E73",  # 2          (bluish green)
    "#F0E442",  # 3          (yellow)
    "#E69F00",  # 4          (orange)
    "#D55E00",  # 5 deep     (vermillion)
    "#CC79A7",  # 6          (reddish purple; Penobscot 7th)
]

# F3 formation names (Alaudah et al. 2019) - stratigraphic order, confirmed
# against the label cube 2026-10-26 (stacking 0->5, class 2 most common,
# class 5 deepest and absent from some contiguous sites).
F3_NAMES = [
    "Upper North Sea",
    "Middle North Sea",
    "Lower North Sea",
    "Rijnland/Chalk",
    "Scruff",
    "Zechstein",
]

# Parihaka (AIcrowd) - 1-indexed raw -> remapped to union rows 6-11. Order per
# findings.md (flagged "to confirm against the challenge docs").
PARIHAKA_NAMES = [
    "Basement/other",
    "Slope mudstone A",
    "Mass transport deposit",
    "Slope mudstone B",
    "Slope valley",
    "Submarine canyon",
]

# Penobscot - 7 facies (union rows 12-18). Label 0 is overburden -> IGNORE (255).
PENOBSCOT_NAMES = [f"Facies {i}" for i in range(1, 8)]

# Union names (6 + 6 + 7 = 19) for multi-survey figures.
UNION_NAMES = F3_NAMES + PARIHAKA_NAMES + PENOBSCOT_NAMES


def facies_cmap(n=6):
    """ListedColormap + BoundaryNorm over the first n palette colors."""
    colors = FACIES_COLORS[:n]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, n, 1), n)
    return cmap, norm


def error_cmap(n=6):
    """White where correct, true-facies color where wrong."""
    colors = ["white"] + FACIES_COLORS[:n]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-1.5, n, 1), n + 1)
    return cmap, norm


# ---------------------------------------------------------------- seismic display
def clip_seismic(section):
    """Symmetric clip at the 2-98 amplitude percentile, per section."""
    lo, hi = np.percentile(section, [2, 98])
    m = max(abs(lo), abs(hi), 1e-8)
    return section, -m, m


def show_seismic(ax, section, cmap="gray"):
    """Grayscale seismic, TRUE aspect ratio, depth increasing downward."""
    img, vmin, vmax = clip_seismic(section)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
              aspect="equal", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def overlay_facies(ax, pred, n=6, alpha=0.45, contour=True):
    """Facies prediction over the current axes (alpha blend + boundary contour)."""
    cmap, _ = facies_cmap(n)
    ax.imshow(pred, cmap=cmap, vmin=-0.5, vmax=n - 0.5, alpha=alpha,
              aspect="equal", interpolation="nearest")
    if contour:
        _contour_boundaries(ax, pred, n)
    return ax


def _contour_boundaries(ax, pred, n):
    import matplotlib.patches as mpatches
    from scipy import ndimage
    # thin dark lines where the facies label changes
    boundary = np.zeros(pred.shape, bool)
    for ax_shift in (0, 1):
        shifted = np.roll(pred, 1, axis=ax_shift)
        boundary |= (shifted != pred)
    ax.contour(boundary.astype(float), levels=[0.5], colors="k",
               linewidths=0.4, alpha=0.6)


def label_panel(ax, text, color="k", size=16, weight="bold", loc="upper left"):
    x = 0.02 if "left" in loc else 0.98
    y = 0.98 if "upper" in loc else 0.02
    ax.text(x, y, text, transform=ax.transAxes, ha="left" if "left" in loc else "right",
            va="top" if "upper" in loc else "bottom", color=color,
            fontsize=size, weight=weight,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))


# ---------------------------------------------------------------- output helpers
OUT = "results/figures/talk"


def save(fig, stem, target_wh=None):
    """Save PNG (300dpi) + PDF + SVG. target_wh=(w,h) inches for the slide ratio."""
    import os
    os.makedirs(OUT, exist_ok=True)
    if target_wh:
        fig.set_size_inches(target_wh)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{OUT}/{stem}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  wrote {OUT}/{stem}.png/.pdf/.svg")


# target aspect ratios (width:height) from the slide placeholders
ASPECT = {
    "full": 3.0, "half": 1.65, "third": 1.07,
    "f4movie": 1.83, "f1map": 1.1, "f1map_unlab": 1.4, "f6": 1.9, "f9": 1.46,
}
