"""F5 (seismic reframe) - "Your data gives you away": BN fingerprint as
crossline sections + BN logs (well-log style), not a PCA scatter.

Traditional interpreters read sections + logs. So: one crossline section
(seismic + facies) per operator, and beside it the BN running-mean signature
drawn as a vertical 'log'. The logs are visibly different per operator -> the
fingerprint. The site-reID number is in FACTS.md, not on the figure.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from fedseismic.config import RunConfig
from fedseismic.privacy.log import load_meta
from talk_figs import style as S

HDD = "/mnt/storage"


def bn_signature(run_dir, client, last_round=20):
    """Concatenated, per-(round,layer)-centered BN running-mean vector for one
    upload -> a 1-D 'log' (same centering as seismic_channel.bn_site_auc)."""
    import glob, os, re
    by_round = {}
    for f in sorted(glob.glob(f"{run_dir}/view/*.npz")):
        m = re.search(r"round_(\d+)_client_(\d+)", f)
        if not m:
            continue
        rnd, cl = int(m.group(1)), int(m.group(2))
        if rnd != last_round:
            continue
        d = np.load(f)
        for k in d.files:
            if k.startswith("bn::") and k.endswith("running_mean"):
                layer = k.split(".running_mean")[0]
                by_round.setdefault((rnd, layer), {})[cl] = np.asarray(d[k], float)
    log = []
    for (rnd, layer) in sorted(by_round):
        dd = by_round[(rnd, layer)]
        if client not in dd:
            continue
        M = np.stack([dd[c] for c in sorted(dd)])
        mu, sd = M.mean(0), M.std(0) + 1e-12
        z = (dd[client] - mu) / sd
        log.append(z)
    # use ONE representative layer (the first-encoder running_mean) so the log is
    # a readable ~64-channel trace, not a dense all-layer scribble
    if not log:
        return np.zeros(1)
    first_layer = sorted(by_round)[0]
    dd = by_round[first_layer]
    M = np.stack([dd[c] for c in sorted(dd)])
    mu, sd = M.mean(0), M.std(0) + 1e-12
    z = (dd[client] - mu) / sd
    return z


def main():
    run = "results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0"
    # view/ may be on the HDD
    run_dir = run if _has_view(run) else run.replace("results/runs", f"{HDD}/results/runs")
    meta = load_meta(run)
    cfg = RunConfig.from_mapping(meta["config"])
    seis = np.load(cfg.train_seismic)
    labs = np.load(cfg.train_labels)
    from fedseismic.data.partition import partition_noniid
    parts = partition_noniid(seis.shape[1], 3)
    ops = "ABC"

    fig = plt.figure(figsize=(12.5, 5.2))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[3.0, 1.2], hspace=0.18, wspace=0.12)

    for i, pr in enumerate(parts):
        # section: richest crossline
        secs = [labs[:, x, :].T for x in pr]
        k = int(np.argmax([len(np.unique(s)) for s in secs]))
        xline = pr[k]
        ax = fig.add_subplot(gs[0, i])
        S.show_seismic(ax, seis[:, xline, :].T)
        S.overlay_facies(ax, labs[:, xline, :].T, n=6, alpha=0.55, contour=True)
        S.label_panel(ax, f"Operator {ops[i]}", color=S.NAVY, size=18)
        if i == 0:
            ax.set_ylabel("depth", fontsize=14)
        # BN log beneath
        lax = fig.add_subplot(gs[1, i])
        sig = bn_signature(run_dir, i)
        y = np.arange(len(sig))
        lax.plot(sig, y, color=S.NAVY, lw=1.4)
        lax.axvline(0, color="#999999", lw=0.8, ls=":")
        lax.set_ylim(len(sig), 0)
        lax.set_yticks([]); lax.tick_params(labelsize=12)
        for sp in ("top", "right", "left"):
            lax.spines[sp].set_visible(False)
        lax.set_title("BN signature", fontsize=13, color=S.NAVY, pad=4)
        if i == 0:
            lax.set_ylabel("BN channel", fontsize=13)
    # facies legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=S.FACIES_COLORS[k], label=S.F3_NAMES[k]) for k in range(6)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=12.5,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    S.save(fig, "f5_sections_bnlogs", (12.5, 5.2))
    print("f5 sections+BN-logs written")


def _has_view(run):
    import os, glob
    return bool(glob.glob(f"{run}/view/*.npz"))


if __name__ == "__main__":
    main()
