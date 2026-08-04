"""Presentation charts for the IMAGE 2026 oral talk.

Design follows the dataviz procedure: form first, color last, emphasis over
categorical. Palette is the validated 2-slot emphasis pair (blue #2a78d6 +
orange #eb6834) — it passes lightness, chroma, CVD separation, normal-vision
floor, and contrast on the light surface used for projection.

Rendered at 200 dpi for a 1920x1080 projector.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'slides')
os.makedirs(OUT, exist_ok=True)

ACCENT = '#2a78d6'   # series 1 — the thing that matters
WARM = '#eb6834'     # series 2 — the contrast
GRAY = '#9a9995'     # de-emphasis
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e4e3df'
SURFACE = '#fcfcfb'

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 17,
    'axes.edgecolor': GRID,
    'axes.labelcolor': INK2,
    'text.color': INK,
    'xtick.color': INK2,
    'ytick.color': INK2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'pdf.fonttype': 42,
})


def _save(fig, name):
    p = os.path.join(OUT, f'{name}.png')
    fig.savefig(p, dpi=200, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print('wrote', p)
    return p


def chart_gap():
    """Emphasis bar: where federated learning sits against centralised."""
    labels = ['Centralised\n(upper bound)', 'IID\nfederated', 'Geographic\nfederated']
    vals = [0.693, 0.686, 0.551]
    colors = [GRAY, GRAY, WARM]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    bars = ax.bar(labels, vals, color=colors, width=0.55, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f'{v:.3f}',
                ha='center', va='bottom', fontsize=21, color=INK, weight='bold')
    ax.set_ylim(0, 0.82)
    ax.set_ylabel('mean IoU')
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.annotate('', xy=(2, 0.693), xytext=(2, 0.551),
                arrowprops=dict(arrowstyle='<->', color=INK2, lw=2))
    ax.text(2.34, 0.62, 'the gap\n0.142', color=INK2, fontsize=17, va='center')
    _save(fig, 'gap')


def _final_c5():
    """Real final-round class-5 IoU across every completed run on disk."""
    import glob
    import re
    vals = []
    for f in glob.glob(os.path.join(os.path.dirname(HERE), 'logs_phase_*', '*.log')):
        last = None
        for line in open(f, errors='ignore'):
            m = re.match(r'\s*(\d+)\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|'
                         r'\s*\[([^\]]+)\]', line)
            if m:
                last = float(m.group(2).split()[5])
        if last is not None:
            vals.append(last)
    return np.array(vals)


def chart_bistability():
    """The scientific finding: two attractors, not a spread. Real runs."""
    v = _final_c5()
    rng = np.random.default_rng(7)
    lost, mid, learned = v[v < 0.005], v[(v >= 0.005) & (v <= 0.15)], v[v > 0.15]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axhspan(0.02, 0.15, color=GRID, alpha=.45, zorder=0)
    for arr, color in ((lost, WARM), (learned, ACCENT), (mid, GRAY)):
        if len(arr):
            ax.scatter(rng.normal(0, 0.14, len(arr)), arr, s=150, color=color,
                       alpha=.85, zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks([])
    ax.set_xlim(-0.72, 1.35)
    ax.set_ylim(-0.025, 0.35)
    ax.set_ylabel('final class-5 IoU')
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.52, 0.0, f'{len(lost)} runs — class lost entirely',
            color=WARM, fontsize=18, va='center', weight='bold')
    ax.text(0.52, 0.245, f'{len(learned)} runs — class learned',
            color=ACCENT, fontsize=18, va='center', weight='bold')
    ax.text(0.52, 0.095, f'only {len(mid)} in between', color=INK2,
            fontsize=17, va='center', style='italic')
    _save(fig, 'bistability')


def chart_rescue():
    """Before/after: the collapsed class recovered at inference."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot([0, 1], [0.008, 0.060], color=ACCENT, lw=3, zorder=2)
    ax.scatter([0, 1], [0.008, 0.060], s=340, zorder=3,
               color=[WARM, ACCENT], edgecolor=SURFACE, linewidth=3)
    ax.text(0, 0.008 - 0.008, '0.008', ha='center', va='top', fontsize=24,
            color=INK, weight='bold')
    ax.text(1, 0.060 + 0.007, '0.060', ha='center', va='bottom', fontsize=24,
            color=INK, weight='bold')
    ax.text(0.5, 0.048, '7.5×', ha='center', fontsize=30, color=ACCENT,
            weight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Collapsed\ncheckpoint', 'After logit adjustment\n(τ = 1.5)'],
                       fontsize=19)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.006, 0.082)
    ax.set_ylabel('class-5 IoU')
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    _save(fig, 'rescue')


def chart_ladder():
    """The ablation ladder — each step isolated, cumulative."""
    steps = ['Baseline\nfederated', '+ aggregation', '+ ensemble', '+ logit\nadjustment']
    vals = [0.5695, 0.5849, 0.6126, 0.6155]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    colors = [GRAY, GRAY, GRAY, ACCENT]
    bars = ax.bar(steps, vals, color=colors, width=0.55, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f'{v:.3f}',
                ha='center', va='bottom', fontsize=20, color=INK, weight='bold')
    ax.axhline(0.693, color=WARM, lw=2.5, ls='--', zorder=4)
    ax.text(3.46, 0.6952, 'centralised  0.693', color=WARM, fontsize=16,
            va='bottom', ha='right')
    ax.set_ylim(0.50, 0.72)
    ax.set_ylabel('mean IoU')
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    _save(fig, 'ladder')


def chart_selection_bias():
    """Our own correction: best-round selection flatters unequally."""
    cfgs = ['Baseline', 'Our best config']
    best = [0.5695, 0.5849]
    final = [0.5514, 0.5421]
    err_b = [0.0104, 0.0085]
    err_f = [0.0121, 0.0753]
    x = np.arange(len(cfgs))
    w = 0.32
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x - w / 2, best, w, yerr=err_b, color=GRAY, zorder=3,
           label='Best round (what we reported)',
           error_kw=dict(ecolor=INK2, lw=2, capsize=7))
    ax.bar(x + w / 2, final, w, yerr=err_f, color=ACCENT, zorder=3,
           label='Final round (unbiased)',
           error_kw=dict(ecolor=INK2, lw=2, capsize=7))
    ax.set_xticks(x)
    ax.set_xticklabels(cfgs, fontsize=19)
    ax.set_ylim(0.40, 0.66)
    ax.set_ylabel('mean IoU')
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=16, loc='upper left')
    ax.annotate('variance was hidden', xy=(1.16, 0.5421), xytext=(1.36, 0.455),
                fontsize=16, color=INK2,
                arrowprops=dict(arrowstyle='->', color=INK2, lw=1.8))
    _save(fig, 'selection_bias')


def chart_tau():
    """State dependence: the right correction depends on the model's bias."""
    tau = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    majority = np.array([.207, .218, .226, .231, .233, .231, .228, .222])
    collapsed = np.array([.008, .020, .032, .045, .052, .056, .060, .058])
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(tau, majority, color=GRAY, lw=3, marker='o', ms=9, zorder=3)
    ax.plot(tau, collapsed, color=ACCENT, lw=3, marker='o', ms=9, zorder=3)
    ax.scatter([0.5], [.226], s=300, facecolor='none', edgecolor=GRAY, lw=3, zorder=4)
    ax.scatter([1.5], [.060], s=300, facecolor='none', edgecolor=ACCENT, lw=3, zorder=4)
    ax.text(0.5, .243, 'best τ = 0.5', color=INK2, fontsize=16, ha='center')
    ax.text(1.5, .073, 'best τ = 1.5', color=ACCENT, fontsize=16, ha='center')
    ax.text(2.03, .222, 'majority-biased\ncheckpoint', color=INK2, fontsize=16, va='center')
    ax.text(2.03, .058, 'collapsed\ncheckpoint', color=ACCENT, fontsize=16, va='center')
    ax.set_xlabel('τ  (strength of the prior correction)')
    ax.set_ylabel('class-5 IoU')
    ax.set_xlim(-0.1, 2.75)
    ax.set_ylim(0, 0.27)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    _save(fig, 'tau')


def chart_norm():
    """Negative result: normalisation is not the lever. Per-seed, no overlap."""
    bn = [0.5736, 0.5638, 0.5715, 0.5678, 0.5675]
    gn = [0.4475, 0.4787, 0.4862, 0.4686]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    rng = np.random.default_rng(3)
    for x, vals, color in ((0, bn, GRAY), (1, gn, WARM)):
        ax.scatter(x + rng.normal(0, 0.045, len(vals)), vals, s=190, color=color,
                   zorder=3, edgecolor=SURFACE, linewidth=2.5)
        ax.hlines(np.mean(vals), x - 0.22, x + 0.22, color=INK, lw=3, zorder=4)
        ax.text(x + 0.30, np.mean(vals), f'{np.mean(vals):.3f}', va='center',
                fontsize=21, color=INK, weight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['BatchNorm\n(what we use)', 'GroupNorm\n(the hypothesis)'],
                       fontsize=19)
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylim(0.42, 0.60)
    ax.set_ylabel('mean IoU')
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.annotate('', xy=(-0.34, 0.5688), xytext=(-0.34, 0.4686),
                arrowprops=dict(arrowstyle='<->', color=INK2, lw=2))
    ax.text(-0.46, 0.519, '−0.100', rotation=90, va='center', ha='center',
            fontsize=17, color=INK2)
    _save(fig, 'norm')


if __name__ == '__main__':
    chart_norm()
    chart_gap()
    chart_bistability()
    chart_rescue()
    chart_ladder()
    chart_selection_bias()
    chart_tau()
