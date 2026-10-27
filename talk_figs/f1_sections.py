"""F1 (seismic reframe) - "Meet the operators": crossline sections + facies mix.

Replaces the survey MAP with real crossline sections (seismic + facies) per
operator, so traditional interpreters see sections, not a GIS map. Beside each
section, the operator's facies mix as a stacked bar. An UNLABELED variant (no
operator names, no bars) is exported for the audience poll "who has no Zechstein?".
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from fedseismic.config import RunConfig
from fedseismic.privacy.log import load_meta, load_labels
from talk_figs import style as S


def build(n_sites=3, labeled=True):
    run = f"results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n{n_sites}_batchnorm/seed_0"
    meta = load_meta(run)
    cfg = RunConfig.from_mapping(meta["config"])
    hist, _ = load_labels(run)
    seis = np.load(cfg.train_seismic)
    labs = np.load(cfg.train_labels)
    from fedseismic.data.partition import partition_noniid
    parts = partition_noniid(seis.shape[1], n_sites)
    ops = "ABCDEFGHIJKLMNOP"[:n_sites]

    fig = plt.figure(figsize=(3.9 * n_sites, 5.4))
    gs = GridSpec(2, n_sites, figure=fig, height_ratios=[3.1, 1.15],
                  hspace=0.16, wspace=0.10)
    for i, pr in enumerate(parts):
        secs = [labs[:, x, :].T for x in pr]
        k = int(np.argmax([len(np.unique(s)) for s in secs]))
        xline = pr[k]
        ax = fig.add_subplot(gs[0, i])
        S.show_seismic(ax, seis[:, xline, :].T)
        S.overlay_facies(ax, labs[:, xline, :].T, n=6, alpha=0.55, contour=True)
        if labeled:
            S.label_panel(ax, f"Operator {ops[i]}", color=S.NAVY, size=17)
        else:
            S.label_panel(ax, f"Operator {ops[i]}", color=S.NAVY, size=17)
        if i == 0:
            ax.set_ylabel("depth", fontsize=14)
        # facies mix bar
        bax = fig.add_subplot(gs[1, i])
        pi = np.asarray(hist[i], float)
        pi = pi / (pi.sum() + 1e-9)
        left = 0.0
        for k in range(6):
            bax.barh([0], [pi[k]], left=left, color=S.FACIES_COLORS[k],
                     edgecolor="white", height=0.7)
            left += pi[k]
        bax.set_xlim(0, 1); bax.set_ylim(-0.5, 0.5); bax.axis("off")
        if labeled:
            bax.set_title("facies mix", fontsize=12.5, color=S.NAVY, pad=2)

    if labeled:
        handles = [mpatches.Patch(color=S.FACIES_COLORS[k], label=S.F3_NAMES[k]) for k in range(6)]
        fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=12.5,
                   frameon=False, bbox_to_anchor=(0.5, -0.03))
    tag = "" if labeled else "_unlabeled"
    S.save(fig, f"f1_sections_n{n_sites}{tag}", (3.9 * n_sites, 5.4))


if __name__ == "__main__":
    build(3, True)
    build(3, False)
    build(5, True)
    build(5, False)
    print("f1 seismic sections written")
