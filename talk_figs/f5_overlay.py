"""F5 fix - overlay all operators' BN signatures on ONE panel so the fingerprint
pops. Sections on top (seismic + facies), and a single combined log panel below
with each operator's BN signature drawn in its own colour -> the divergence is
the point (the per-operator logs looked too similar side by side).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from fedseismic.config import RunConfig
from fedseismic.privacy.log import load_meta
from talk_figs import style as S
from talk_figs.f5_sections_bnlogs import bn_signature, _has_view

SITE_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
HDD = "/mnt/storage"


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

    fig = plt.figure(figsize=(12.5, 5.6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[3.0, 1.4], hspace=0.16, wspace=0.10)

    sigs = []
    for i, pr in enumerate(parts):
        secs = [labs[:, x, :].T for x in pr]
        k = int(np.argmax([len(np.unique(s)) for s in secs]))
        xline = pr[k]
        ax = fig.add_subplot(gs[0, i])
        S.show_seismic(ax, seis[:, xline, :].T)
        S.overlay_facies(ax, labs[:, xline, :].T, n=6, alpha=0.55, contour=True)
        S.label_panel(ax, f"Operator {ops[i]}", color=S.NAVY, size=18)
        if i == 0:
            ax.set_ylabel("depth", fontsize=14)
        sigs.append(bn_signature(run_dir, i))

    # combined BN-signature panel spanning all 3 columns
    lax = fig.add_subplot(gs[1, :])
    for i, sig in enumerate(sigs):
        y = np.arange(len(sig))
        lax.plot(sig, y, color=SITE_COLORS[i], lw=1.5, label=f"Operator {ops[i]}")
    lax.axvline(0, color="#666666", lw=0.9, ls=":")
    lax.set_ylim(max(len(s) for s in sigs), 0)
    lax.set_yticks([]); lax.tick_params(labelsize=12)
    for sp in ("top", "right", "left"):
        lax.spines[sp].set_visible(False)
    lax.set_xlabel("BN running-mean (z-score per channel)", fontsize=13)
    lax.set_title("BN signatures diverge -> each operator is identifiable",
                  fontsize=13.5, color=S.NAVY, pad=5)
    lax.legend(loc="lower right", fontsize=12, frameon=True, framealpha=0.9)

    S.save(fig, "f5_fingerprint_overlay", (12.5, 5.6))
    print("f5 overlay written")


if __name__ == "__main__":
    main()
