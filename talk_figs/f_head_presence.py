"""Real vs INFERRED presence, seismic-first.

Per operator: the crossline section (seismic + facies) beside a stratigraphic
presence column for REAL and for INFERRED (from the plain-FedAvg head, no
SecAgg). Held facies = filled band; lacking = empty + X. Reads as sections +
strat columns, so it looks like geology, not a matrix.
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from fedseismic.config import RunConfig
from fedseismic.experiment import build_federated_run
from fedseismic.privacy.log import load_meta, load_labels
from talk_figs import style as S

RUN = "results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0"


def inferred_presence(run_dir, n=3, classes=6, last=20):
    out = []
    for c in range(n):
        f = glob.glob(f"{run_dir}/head_outc/round_{last:03d}_client_{c:03d}.npz")
        bias = np.load(f[0])["bias_delta"].reshape(-1)[:classes] if f else np.zeros(classes)
        thr = 0.5 * (bias.max() + bias.min())
        out.append(bias > thr)
    return out


def presence_col(ax, present, x):
    """vertical stratigraphic column of 6 facies bands (top=shallow)."""
    for k in range(6):
        y = 5 - k
        if present[k]:
            ax.add_patch(Rectangle((x - 0.42, y - 0.42), 0.84, 0.84,
                                   facecolor=S.FACIES_COLORS[k], edgecolor="white", lw=1.4))
        else:
            ax.add_patch(Rectangle((x - 0.42, y - 0.42), 0.84, 0.84,
                                   facecolor="white", edgecolor="#C23B22", lw=1.8))
            ax.plot([x - 0.24, x + 0.24], [y - 0.24, y + 0.24], color="#C23B22", lw=2.2)
            ax.plot([x - 0.24, x + 0.24], [y + 0.24, y - 0.24], color="#C23B22", lw=2.2)


def main():
    meta = load_meta(RUN)
    cfg = RunConfig.from_mapping(meta["config"])
    data = build_federated_run(cfg, 0)
    seis = np.load(cfg.train_seismic)
    labs = np.load(cfg.train_labels)
    hist, _ = load_labels(RUN)
    from fedseismic.data.partition import partition_noniid
    parts = partition_noniid(seis.shape[1], 3)
    names = S.F3_NAMES
    inf = inferred_presence(RUN)

    fig = plt.figure(figsize=(13.5, 5.6))
    for i in range(3):
        # seismic + facies section (left)
        ax = fig.add_subplot(2, 3, i + 1)
        pr = parts[i]; xline = pr[len(pr) // 2]
        S.show_seismic(ax, seis[:, xline, :].T)
        S.overlay_facies(ax, labs[:, xline, :].T, n=6, alpha=0.55, contour=True)
        ax.set_title(f"Operator {'ABC'[i]}", fontsize=15, color=S.NAVY, pad=6)
        if i == 0:
            ax.set_ylabel("depth", fontsize=12)

        # real + inferred presence columns (bottom)
        axb = fig.add_subplot(2, 3, i + 4)
        real_p = [hist[i][k] > 0 for k in range(6)]
        presence_col(axb, real_p, 0.75)
        presence_col(axb, [bool(inf[i][k]) for k in range(6)], 1.75)
        axb.text(0.75, -0.95, "real", ha="center", fontsize=12, color=S.NAVY, fontweight="bold")
        axb.text(1.75, -0.95, "inferred", ha="center", fontsize=12, color=S.NAVY)
        axb.set_xlim(-0.2, 2.7); axb.set_ylim(-1.5, 5.9)
        axb.axis("off")

    fig.text(0.02, 0.30, "← real\nvs\ninferred →", fontsize=11, color=S.NAVY)
    handles = [Patch(color=S.FACIES_COLORS[k], label=names[k]) for k in range(6)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    S.save(fig, "f_head_presence", (13.5, 5.6))
    print("f_head_presence (seismic + real vs inferred) written")


if __name__ == "__main__":
    main()
