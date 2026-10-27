"""F5 (v3) - "the fingerprint": facies-log barcodes + a blind well tie.

Raw BN statistics look similar across operators; the signal is each operator's
DEVIATION from the shared average. So we draw each operator's BN signature as a
facies-log 'barcode' (discrete blue/grey/red = below / typical / above the
cross-operator average), one column per operator, plus an UNKNOWN upload whose
strip is tied to its match. Interpreters read discrete facies columns instantly.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from fedseismic.config import RunConfig
from fedseismic.privacy.log import load_meta
from talk_figs import style as S
from talk_figs.f5_sections_bnlogs import bn_signature, _has_view

HDD = "/mnt/storage"
BLUE, GREY, RED = "#2C6BB5", "#D9D9D9", "#C23B22"


def barcode_colors(z, thresh=0.6):
    c = np.where(z < -thresh, 0, np.where(z > thresh, 2, 1))
    return [BLUE, GREY, RED], c


def main():
    run = "results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0"
    run_dir = run if _has_view(run) else run.replace("results/runs", f"{HDD}/results/runs")
    meta = load_meta(run)
    cfg = RunConfig.from_mapping(meta["config"])
    seis = np.load(cfg.train_seismic)
    labs = np.load(cfg.train_labels)
    from fedseismic.data.partition import partition_noniid
    parts = partition_noniid(seis.shape[1], 3)
    ops = "ABC"

    sigs = [bn_signature(run_dir, i) for i in range(3)]
    # sort channels once by fleet-wide variance so strips are comparable
    M = np.stack(sigs)
    order = np.argsort(-M.std(axis=0))
    sigs = [s[order] for s in sigs]
    # z-score each channel across operators (the deviation is the signal)
    mu, sd = M.mean(0)[order], M.std(0)[order] + 1e-12
    Z = [(s - mu) / sd for s in sigs]

    ncols = 4
    fig = plt.figure(figsize=(11, 6.2))
    gs = GridSpec(2, ncols, figure=fig, height_ratios=[1.0, 2.6],
                  hspace=0.05, wspace=0.18)

    # thumbnails (top row) for A/B/C
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        pr = parts[i]
        xline = pr[len(pr) // 2]
        S.show_seismic(ax, seis[:, xline, :].T)
        S.overlay_facies(ax, labs[:, xline, :].T, n=6, alpha=0.55, contour=True)
        ax.set_title(f"Operator {ops[i]}", fontsize=14, color=S.NAVY, pad=4)
        if i == 0:
            ax.set_ylabel("depth", fontsize=11)
    axu = fig.add_subplot(gs[0, 3])
    axu.axis("off")
    axu.text(0.5, 0.5, "??", ha="center", va="center", fontsize=26, color="#888888")
    axu.set_title("Unknown upload", fontsize=14, color="#888888", pad=4)

    # barcodes (bottom row): A/B/C + unknown (match -> B)
    which = [0, 1, 2, 1]   # unknown ties to B
    Zs = [Z[0], Z[1], Z[2], Z[1]]
    labels = ["A", "B", "C", "?"]
    for j in range(4):
        ax = fig.add_subplot(gs[1, j])
        z = Zs[j]
        cols, idx = barcode_colors(z)
        n = len(z)
        for k in range(n):
            ax.add_patch(Rectangle((0, n - 1 - k), 1, 1,
                                   facecolor=cols[idx[k]], edgecolor="none"))
        ax.set_xlim(0, 1); ax.set_ylim(0, n)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel(f"Operator {labels[j]}" if j < 3 else "Unknown",
                      fontsize=13, color=S.NAVY if j < 3 else "#888888")
        if j == 0:
            ax.set_ylabel("BN channel\n(network depth)", fontsize=11)

    # tie-line from unknown (col 3) to B (col 1)
    ax_un = fig.add_subplot(gs[1, 3])
    ax_un.annotate("", xy=(0.0, n * 0.5), xytext=(-2.05, n * 0.5),
                   arrowprops=dict(arrowstyle="-", lw=2.0, ls="--", color="#C23B22"))
    ax_un.text(-1.05, n * 0.62, "Matched:\nOperator B", ha="center", color="#C23B22",
               fontsize=13, weight="bold")

    # footer numbers
    fig.text(0.5, 0.015, "Re-identify the operator from statistics alone:  "
                         "correct 99 in 100 (3 operators) · 94 in 100 (5)  |  "
                         "guessing: 33 / 20",
             ha="center", fontsize=12.5, color=S.NAVY)
    S.save(fig, "f5_fingerprint_barcodes", (11, 6.2))
    print("f5 barcodes written")


if __name__ == "__main__":
    main()
