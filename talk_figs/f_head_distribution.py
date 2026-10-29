"""Right-hand slide figure (v2): REAL vs INFERRED facies distribution.

Top bar = the true distribution (ground truth). Bottom bar = the distribution
inferred from the plain-FedAvg classification-head update (bias deltas: a held
class pulls its bias down, a missing one leaves it up). Showing they match is
the point: the shared head already hands us each site's distribution. Exact
counts do not calibrate on F3, so the inferred bar is presence + relative mix.
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from fedseismic.privacy.log import load_labels
from talk_figs import style as S

RUN = "results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0"


def inferred_mix(run_dir, n=3, classes=6, last=20):
    """Per-site distribution from the head bias deltas: a held class pulls its
    bias down, so count_k ~ -bias_k (clipped at 0), normalised."""
    out = []
    for c in range(n):
        f = glob.glob(f"{run_dir}/head_outc/round_{last:03d}_client_{c:03d}.npz")
        if not f:
            out.append(np.full(classes, np.nan))
            continue
        bias = np.load(f[0])["bias_delta"].reshape(-1)[:classes]
        est = np.clip(bias - np.median(bias), 0, None)  # held class -> positive bias update
        if est.sum() <= 0:
            est = np.full(classes, 1.0 / classes)
        out.append(est / est.sum())
    return out


def main():
    hist, _ = load_labels(RUN)
    names = S.F3_NAMES
    ops = ["Operator A", "Operator B", "Operator C"]
    real = [np.asarray(hist[i], float) / (np.asarray(hist[i], float).sum() + 1e-9) for i in range(3)]
    inferred = inferred_mix(RUN)

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    h = 0.34
    for i in range(3):
        ybase = 2 - i
        for j, (mix, tag) in enumerate([(real[i], "real"), (inferred[i], "inferred")]):
            y = ybase + (0.22 if j == 0 else -0.22)
            left = 0.0
            for k in range(6):
                if not np.isfinite(mix[k]) or mix[k] <= 0:
                    continue
                ax.barh([y], [mix[k]], left=left, height=h,
                        color=S.FACIES_COLORS[k], edgecolor="white", linewidth=1.0,
                        alpha=1.0 if j == 0 else 0.55)
                left += mix[k]
            if i == 0:
                ax.text(1.02, y, tag, va="center", fontsize=11, color=S.NAVY,
                        fontweight="bold" if j == 0 else "normal")

    ax.set_yticks([2, 1, 0]); ax.set_yticklabels(ops, fontsize=15, color=S.NAVY)
    ax.set_ylim(-0.6, 2.7)
    ax.set_xlim(0, 1.16)
    ax.set_xlabel("share of facies  (real = ground truth; inferred = from the head)", fontsize=12, color=S.NAVY)
    ax.tick_params(length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    handles = [Patch(color=S.FACIES_COLORS[k], label=names[k]) for k in range(6)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.16),
              ncol=6, fontsize=9, frameon=False)
    S.save(fig, "f_head_distribution", (8.8, 4.6))
    print("f_head_distribution (real vs inferred) written")


if __name__ == "__main__":
    main()
