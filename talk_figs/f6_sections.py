"""F6 (seismic reframe) - "Who has the Zechstein?" told in crossline sections.

Traditional interpreters want the seismic scan with the facies overlay, not a
rank heatmap. So instead of a site x facies matrix we show one REAL crossline
section per operator (grayscale seismic + facies overlay), so the interpreter
sees that Operator A has no Zechstein while Operator C does. A small caption
ties it to the privacy point (the head update reveals this).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from fedseismic.config import RunConfig
from fedseismic.experiment import build_federated_run
from fedseismic.privacy.log import load_meta, load_labels
from talk_figs import style as S


def pick_representative(labels_sec, cls=5, prefer="present"):
    """crossline index (within the block) whose section best shows class `cls`."""
    # labels_sec: list of (depth, inline) arrays per crossline in the block
    counts = np.array([(sec == cls).sum() for sec in labels_sec])
    if prefer == "present":
        return int(np.argmax(counts))
    return int(np.argmin(counts))


def main():
    run = "results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n5_batchnorm/seed_0"
    meta = load_meta(run)
    cfg = RunConfig.from_mapping(meta["config"])
    hist, rare = load_labels(run)

    import numpy as _np
    seis = _np.load(cfg.train_seismic)
    labs = _np.load(cfg.train_labels)
    H, W, D = seis.shape

    from fedseismic.data.partition import partition_noniid
    parts = partition_noniid(W, cfg.num_clients)  # contiguous crossline blocks

    # choose one representative crossline per operator
    n_sites = cfg.num_clients
    ops = "ABCDEFGHIJKLMNOP"[:n_sites]
    chosen = []
    for i, pr in enumerate(parts):
        secs = [labs[:, x, :].T for x in pr]  # (depth, inline) each
        # pick the crossline that best shows the facies mix (most total facies variety)
        richness = [len(np.unique(s)) for s in secs]
        k = int(np.argmax(richness))
        chosen.append((pr[k], secs[k], i))

    fig = plt.figure(figsize=(4.0 * n_sites, 4.0))
    gs = GridSpec(1, n_sites, figure=fig, wspace=0.04)

    for idx, (xline, lab_sec, i) in enumerate(chosen):
        ax = fig.add_subplot(gs[0, idx])
        sec = seis[:, xline, :].T  # (depth, inline)
        S.show_seismic(ax, sec)
        S.overlay_facies(ax, lab_sec, n=6, alpha=0.55, contour=True)
        has_z = bool((lab_sec == 5).any())
        S.label_panel(ax, f"Operator {ops[i]}", color=S.NAVY, size=18)
        # mark Zechstein presence/absence clearly
        tag = "has Zechstein" if has_z else "NO Zechstein"
        col = S.FACIES_COLORS[5] if has_z else "#000000"
        ax.set_title(tag, color=col, fontsize=15, weight="bold", pad=8)
        if idx == 0:
            ax.set_ylabel("depth", fontsize=14)

    # facies legend (formation names) under the whole strip
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=S.FACIES_COLORS[k], label=S.F3_NAMES[k]) for k in range(6)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=13,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    S.save(fig, "f6_sections_n5", (4.0 * n_sites, 4.0))

    # also a version for N=3 (the poll-friendly one: Operator A has no Zechstein)
    run3 = "results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0"
    meta3 = load_meta(run3)
    cfg3 = RunConfig.from_mapping(meta3["config"])
    parts3 = partition_noniid(W, cfg3.num_clients)
    labs3 = _np.load(cfg3.train_labels)
    chosen3 = []
    for i, pr in enumerate(parts3):
        secs = [labs3[:, x, :].T for x in pr]
        richness = [len(np.unique(s)) for s in secs]
        k = int(np.argmax(richness))
        chosen3.append((pr[k], secs[k], i))
    fig3 = plt.figure(figsize=(4.2 * 3, 4.0))
    gs3 = GridSpec(1, 3, figure=fig3, wspace=0.04)
    ops3 = "ABC"
    for idx, (xline, lab_sec, i) in enumerate(chosen3):
        ax = fig3.add_subplot(gs3[0, idx])
        sec = seis[:, xline, :].T
        S.show_seismic(ax, sec)
        S.overlay_facies(ax, lab_sec, n=6, alpha=0.55, contour=True)
        has_z = bool((lab_sec == 5).any())
        S.label_panel(ax, f"Operator {ops3[i]}", color=S.NAVY, size=18)
        tag = "has Zechstein" if has_z else "NO Zechstein"
        col = S.FACIES_COLORS[5] if has_z else "#000000"
        ax.set_title(tag, color=col, fontsize=15, weight="bold", pad=8)
        if idx == 0:
            ax.set_ylabel("depth", fontsize=14)
    handles3 = [mpatches.Patch(color=S.FACIES_COLORS[k], label=S.F3_NAMES[k]) for k in range(6)]
    fig3.legend(handles=handles3, loc="lower center", ncol=6, fontsize=13,
                frameon=False, bbox_to_anchor=(0.5, -0.02))
    S.save(fig3, "f6_sections_n3", (4.2 * 3, 4.0))
    print("f6 seismic sections written")


if __name__ == "__main__":
    main()
