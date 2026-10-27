"""F7 / F8 / F9 - illustrative schematics (no GPU, no run data).

F7 "Silence is a signal"  : head rows over rounds; held facies move, the absent
                            facies row stays flat.
F8 "Sealed totals"        : three operators -> one sealed sum; amounts blurred,
                            presence dots stay lit.
F9 "What stays home"      : UNet sketch; shared body vs local BN + head.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from talk_figs import style as S


def f7_silence(seed=0):
    rng = np.random.default_rng(seed)
    rounds = np.arange(1, 11)
    names = S.F3_NAMES
    fig, ax = plt.subplots(figsize=(8, 6))
    absent = 5  # Zechstein absent at this site
    label_y = []
    for k in range(6):
        base = 5.0 + 1.4 * k
        if k == absent:
            y = np.full_like(rounds, base, dtype=float)  # dead flat
        else:
            y = base + np.cumsum(rng.normal(0, 0.22, len(rounds)))  # drifts
        lw = 3.0 if k != absent else 2.2
        ax.plot(rounds, y, color=S.FACIES_COLORS[k], lw=lw, solid_capstyle="round")
        end_y = y[-1]
        label_y.append((end_y, k))
    # spread labels vertically so they never collide
    label_y.sort()
    min_gap = 0.62
    for i in range(1, len(label_y)):
        if label_y[i][0] - label_y[i - 1][0] < min_gap:
            label_y[i] = (label_y[i - 1][0] + min_gap, label_y[i][1])
    for ly, k in label_y:
        ax.text(rounds[-1] + 0.15, ly, names[k], color=S.FACIES_COLORS[k],
                va="center", fontsize=14, weight="bold")
    ax.plot(rounds, np.full_like(rounds, 5.0 + 1.4 * absent), color=S.FACIES_COLORS[absent],
            lw=1.0, ls=":", alpha=0.5)
    ax.text(rounds[0], 5.0 + 1.4 * absent - 0.55, "absent here  ->  silent",
            color=S.FACIES_COLORS[absent], fontsize=14, weight="bold")
    ax.set_xlabel("round"); ax.set_ylabel("head row update")
    ax.set_yticks([]); ax.set_xticks([1, 5, 10])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    S.save(fig, "f7_silence", (8, 6))


def f8_sealed_sum(seed=0):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    site_cols = ["#1b9e77", "#7570b3", "#e7298a"]  # operator colors (not facies hues)
    labels = ["Operator A", "Operator B", "Operator C"]
    mixes = [[.3, .2, .35, .1, .05, 0], [.25, .3, .2, .05, .15, .05],
             [.2, .25, .3, .15, .05, .05]]
    for i in range(3):
        y0 = 7.6 - i * 1.25
        ax.add_patch(FancyBboxPatch((0.6, y0), 2.5, 0.9, boxstyle="round,pad=0.02",
                                    fc=site_cols[i], ec="none", alpha=0.9))
        ax.text(1.85, y0 + 0.45, labels[i], ha="center", va="center",
                color="white", fontsize=14, weight="bold")
        ax.add_patch(FancyArrowPatch((3.2, y0 + 0.45), (5.4, 5.2),
                                     arrowstyle="-|>", mutation_scale=16,
                                     color=site_cols[i], lw=2, alpha=0.8))
    # sealed box
    ax.add_patch(FancyBboxPatch((5.4, 3.4), 3.9, 3.4, boxstyle="round,pad=0.03",
                                fc=S.NAVY, ec=S.GOLD, lw=2.5))
    ax.text(7.35, 6.45, "one sealed sum", ha="center", color="white",
            fontsize=15, weight="bold")
    # presence dots (lit) + blurred amount bars
    fac_present = [1, 1, 1, 1, 1, 0]
    for k in range(6):
        y = 5.75 - k * 0.42
        dot_c = S.FACIES_COLORS[k] if fac_present[k] else "#888888"
        ax.scatter([6.05], [y], s=95, color=dot_c, zorder=3)
        # blurred amount bar
        bar = 1.3 * (0.4 + 0.6 * rng.random()) if fac_present[k] else 0.12
        ax.add_patch(Rectangle((6.35, y - 0.12), bar, 0.24, fc="white",
                               ec="none", alpha=0.45 if fac_present[k] else 0.18))
    ax.text(7.35, 3.05, "amounts hidden   |   presence visible", ha="center",
            color=S.NAVY, fontsize=14, weight="bold")
    S.save(fig, "f8_sealed_sum", (8.5, 6))


def f9_unet():
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 10); ax.axis("off")

    def block(x, y, w, h, fc, ec, text, tc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=tc, fontsize=12.5, weight="bold")

    shared = "#7a90a8"; local = S.GOLD
    # encoder
    for i in range(4):
        block(1.2, 7.6 - i * 1.7, 1.7, 1.15, shared, S.NAVY, f"Conv {4-i}")
    block(1.2, 1.0, 1.7, 1.2, shared, S.NAVY, "Conv 1")
    # bottleneck
    block(5.2, 3.7, 1.8, 1.5, shared, S.NAVY, "Bottleneck")
    # decoder
    for i in range(4):
        block(9.0, 1.0 + i * 1.7, 1.7, 1.15, shared, S.NAVY, f"Up {i+1}")
    # BN layers (local) as small gold ticks along encoder/decoder
    for i in range(5):
        ax.add_patch(Rectangle((3.05, 7.65 - i * 1.7 + 0.25), 0.5, 0.65, fc=local, ec=S.NAVY, lw=1))
        ax.add_patch(Rectangle((8.35, 1.25 + i * 1.7 + 0.25), 0.5, 0.65, fc=local, ec=S.NAVY, lw=1))
    ax.text(3.3, 9.5, "BN (local)", ha="center", color=S.NAVY, fontsize=12.5, weight="bold")
    ax.text(8.6, 9.5, "BN (local)", ha="center", color=S.NAVY, fontsize=12.5, weight="bold")
    # skips
    for i in range(4):
        ax.add_patch(FancyArrowPatch((3.05, 8.1 - i * 1.7), (8.95, 1.6 + i * 1.7),
                                     arrowstyle="-", lw=1.1, ls="--", color="#b0b0b0",
                                     connectionstyle="arc3,rad=0.12"))
    ax.text(6.1, 9.4, "skip connections", ha="center", color="#888888", fontsize=12.5)
    # head (local)
    block(11.6, 4.1, 1.2, 1.7, local, S.NAVY, "Head", tc=S.NAVY)
    ax.add_patch(FancyArrowPatch((10.75, 4.95), (11.55, 4.95), arrowstyle="-|>",
                                 mutation_scale=14, color=S.NAVY, lw=2))
    ax.text(12.2, 3.6, "1x1 seg\n(local)", ha="center", color=S.NAVY, fontsize=12.5, weight="bold")
    # legend
    ax.add_patch(Rectangle((1.2, 0.15), 0.55, 0.5, fc=shared, ec=S.NAVY))
    ax.text(1.95, 0.4, "shared by all operators", va="center", fontsize=14)
    ax.add_patch(Rectangle((7.4, 0.15), 0.55, 0.5, fc=local, ec=S.NAVY))
    ax.text(8.15, 0.4, "kept at each site", va="center", fontsize=14)
    S.save(fig, "f9_unet_keep_home", (9, 6.2))


if __name__ == "__main__":
    f7_silence()
    f8_sealed_sum()
    f9_unet()
    print("schematics done")
