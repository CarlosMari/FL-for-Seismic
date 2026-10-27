"""F9 (redesigned) - "What stays home": clean U-Net diagram.

A proper U: encoder down the left, decoder up the right, bottleneck at the
base, skip connections arcing across. Shared body = conv blocks (blue-grey);
kept-at-site = BatchNorm + the 1x1 head (gold). Short labels, no overflow.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from talk_figs import style as S


def block(ax, x, y, w, h, fc, ec, text, tc="white", fs=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                fc=fc, ec=ec, lw=1.6, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, weight="bold")


def main():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 14); ax.set_ylim(0, 12); ax.axis("off")

    shared = "#7a90a8"
    local = S.GOLD
    enc_x, dec_x = 1.2, 9.6
    w, h = 2.7, 1.5
    ys = [10.0, 7.7, 5.4, 3.1]  # encoder top->bottom (Enc1..Enc4)

    # encoder (shared) with a local BN badge on each
    enc_centers = []
    for i, y in enumerate(ys):
        block(ax, enc_x, y, w, h, shared, S.NAVY, f"Conv {i+1}")
        ax.add_patch(FancyBboxPatch((enc_x + w, y + 0.25), 0.7, h - 0.5,
                                    boxstyle="round,pad=0.06", fc=local, ec=S.NAVY, lw=1.3))
        ax.text(enc_x + w + 0.35, y + h / 2, "BN", ha="center", va="center",
                color=S.NAVY, fontsize=10, weight="bold")
        enc_centers.append((enc_x + w / 2, y + h / 2))

    # bottleneck (shared) at the base
    bn_x, bn_y = 5.4, 1.2
    block(ax, bn_x, bn_y, 3.2, h, shared, S.NAVY, "Bottleneck", fs=13)

    # decoder (shared) up the right, with a local BN badge on each
    dec_centers = []
    for i, y in enumerate(ys[::-1]):  # Dec1 top ... Dec4 bottom
        block(ax, dec_x, y, w, h, shared, S.NAVY, f"Up {i+1}")
        ax.add_patch(FancyBboxPatch((dec_x - 0.7, y + 0.25), 0.7, h - 0.5,
                                    boxstyle="round,pad=0.06", fc=local, ec=S.NAVY, lw=1.3))
        ax.text(dec_x - 0.35, y + h / 2, "BN", ha="center", va="center",
                color=S.NAVY, fontsize=10, weight="bold")
        dec_centers.append((dec_x + w / 2, y + h / 2))

    # skip connections: encoder level i -> decoder level i (arcing over the middle)
    for (ex, ey), (dx, dy) in zip(enc_centers, dec_centers[::-1]):
        ax.add_patch(FancyArrowPatch((ex + 1.55, ey), (dx - 1.55, dy),
                                     arrowstyle="-", lw=1.2, ls="--", color="#b7b7b7",
                                     connectionstyle="arc3,rad=0.18"))

    # encoder -> bottleneck -> decoder flow arrows
    ax.add_patch(FancyArrowPatch((enc_centers[-1][0] + 1.55, enc_centers[-1][1]),
                                 (bn_x + 0.2, bn_y + h / 2),
                                 arrowstyle="-|>", mutation_scale=15, color=S.NAVY, lw=2))
    ax.add_patch(FancyArrowPatch((bn_x + 3.0, bn_y + h / 2),
                                 (dec_centers[-1][0] - 1.55, dec_centers[-1][1]),
                                 arrowstyle="-|>", mutation_scale=15, color=S.NAVY, lw=2))

    # head (local) off the top decoder
    block(ax, 12.9, ys[0], 1.2, h, local, S.NAVY, "Head", tc=S.NAVY, fs=12)
    ax.add_patch(FancyArrowPatch((dec_x + w, ys[0] + h / 2), (12.85, ys[0] + h / 2),
                                 arrowstyle="-|>", mutation_scale=15, color=S.NAVY, lw=2))
    ax.text(13.5, ys[0] - 0.55, "1x1 seg", ha="center", color=S.NAVY, fontsize=11, weight="bold")

    ax.text(2.55, 11.3, "Encoder", ha="center", color=S.NAVY, fontsize=13, weight="bold")
    ax.text(11.0, 11.3, "Decoder", ha="center", color=S.NAVY, fontsize=13, weight="bold")
    ax.text(7.0, 2.95, "skip", ha="center", color="#999999", fontsize=11)

    # legend
    ax.add_patch(FancyBboxPatch((1.2, 0.15), 1.1, 0.7, boxstyle="round,pad=0.1",
                                fc=shared, ec=S.NAVY, lw=1.4))
    ax.text(2.6, 0.5, "shared by all operators", va="center", fontsize=13.5)
    ax.add_patch(FancyBboxPatch((8.2, 0.15), 1.1, 0.7, boxstyle="round,pad=0.1",
                                fc=local, ec=S.NAVY, lw=1.4))
    ax.text(9.6, 0.5, "kept at each site (BN + head)", va="center", fontsize=13.5)

    S.save(fig, "f9_unet_keep_home", (10, 7))
    print("f9 redesigned")


if __name__ == "__main__":
    main()
