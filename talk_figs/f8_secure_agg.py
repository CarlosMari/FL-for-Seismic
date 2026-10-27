"""F8 (v2) - "sealed totals": before/after of what the SERVER sees.

Drop the sealed-box metaphor. Top row: without secure aggregation the server
sees a per-facies bar SPLIT BY OPERATOR (who has what, how much). Bottom row:
with secure aggregation the same totals are one grey bar - who/how-much are
hidden, but a rare facies (salt/Zechstein) is still visible. Use the same
operator colours as F5 so the audience links them.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from talk_figs import style as S

OP_COLORS = ["#2C6BB5", "#E08214", "#3B8C6E"]  # A/B/C (match F5-ish, distinct)
FACIES = S.F3_NAMES
MIX = np.array([
    [.28, .12, .34, .14, .08, .04],   # A
    [.22, .16, .30, .16, .10, .06],   # B
    [.18, .14, .32, .14, .10, .12],   # C  (has the salt/Zechstein)
])
TOTAL = MIX.sum(0)


def main():
    fig = plt.figure(figsize=(11.5, 6.4))
    gs = GridSpec(2, 1, figure=fig, hspace=0.22)
    x = np.arange(len(FACIES))
    w = 0.62

    # TOP: without secure aggregation (split by operator)
    ax = fig.add_subplot(gs[0])
    bottom = np.zeros(len(FACIES))
    for i in range(3):
        ax.bar(x, MIX[i], w, bottom=bottom, color=OP_COLORS[i],
               edgecolor="white", linewidth=0.6)
        bottom += MIX[i]
    ax.set_xticks(x); ax.set_xticklabels(FACIES, fontsize=11, rotation=20, ha="right")
    ax.set_ylabel("share of update", fontsize=12)
    ax.set_title("Without secure aggregation  →  server sees WHO has WHAT, and HOW MUCH",
                 fontsize=13.5, color=S.NAVY, pad=6)
    ax.set_ylim(0, TOTAL.max() * 1.28)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # BOTTOM: with secure aggregation (one grey bar)
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(x, TOTAL, w, color="#B7B7B7", edgecolor="white", linewidth=0.6)
    ax2.set_xticks(x); ax2.set_xticklabels(FACIES, fontsize=11, rotation=20, ha="right")
    ax2.set_ylabel("share of update", fontsize=12)
    ax2.set_title("With secure aggregation  →  server sees one total",
                  fontsize=13.5, color=S.NAVY, pad=6)
    ax2.set_ylim(0, TOTAL.max() * 1.28)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    # callouts
    ax2.annotate("hidden: who contributed,\nand how much each",
                 xy=(1.0, TOTAL.max() * 1.12), fontsize=12, color="#2E7D32",
                 ha="center", weight="bold")
    ax2.annotate("still visible:\nsomeone has salt",
                 xy=(5, TOTAL[5]), xytext=(4.15, TOTAL.max() * 1.12),
                 fontsize=12, color="#C62828", weight="bold", ha="center",
                 arrowprops=dict(arrowstyle="-|>", color="#C62828", lw=1.6))
    ax2.text(0.5, -0.20, "a rare facies is still a fingerprint:  if only one partner could "
                         "have salt, 'someone' means them.",
             transform=ax2.transAxes, ha="center", fontsize=10.5, color="#888888")

    S.save(fig, "f8_secure_agg", (11.5, 6.4))
    print("f8 before/after written")


if __name__ == "__main__":
    main()
