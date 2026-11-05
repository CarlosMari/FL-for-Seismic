"""Fig 6.3: centered round-20 head-bias change next to the true training share."""
from thesis_ch6_common import *

import matplotlib.patheffects as pe

from f6_xray import balanced_accuracy, column_ranks, job_pooled_within
from fedseismic.privacy.log import load_labels
from fedseismic.privacy.seismic_channel import ABSENT, outc_channels

RUN_ROOT = ROOT / "results/runs/seismic_f3"
JOB = "fedavg_noniid_a0.1_lNA_dna_n{}_batchnorm"
SHORT = ["U. North Sea", "M. North Sea", "L. North Sea", "Rijnland/Chalk", "Scruff", "Zechstein"]
CELL_H = 0.26


def load(n):
    run = RUN_ROOT / JOB.format(n) / "seed_0"
    hist, _ = load_labels(run)
    pis = np.stack([hist[c] for c in sorted(hist)])
    _, biases = outc_channels(run)
    return np.stack([biases[c] for c in sorted(biases)]), pis


def heat(ax, values, ranks, absent, vmax=None, tint=False):
    sites, facies = values.shape
    if tint:
        img = np.ones((sites, facies, 3))
        for k in range(facies):
            top = values[:, k].max()
            for i in range(sites):
                if not absent[i, k]:
                    img[i, k] = 1 - 0.65 * (values[i, k] / top) * (1 - np.array([0.25, 0.45, 0.65]))
        ax.imshow(img, aspect="auto")
    else:
        ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    halo = [pe.withStroke(linewidth=1.5, foreground="white")]
    for i in range(sites):
        for k in range(facies):
            if absent[i, k]:
                dark = not tint and abs(values[i, k]) > 0.6 * vmax
                ax.plot([k - 0.17, k + 0.17], [i, i], color="white" if dark else "k", lw=1.1,
                        solid_capstyle="butt", zorder=6)
                ax.add_patch(mpatches.Rectangle((k - 0.45, i - 0.45), 0.9, 0.9, fill=False,
                                                ec="white", lw=2.2, zorder=4))
                ax.add_patch(mpatches.Rectangle((k - 0.45, i - 0.45), 0.9, 0.9, fill=False,
                                                ec="k", lw=1.0, zorder=5))
            else:
                ax.text(k, i, str(ranks[i, k]), ha="center", va="center", fontsize=8, path_effects=halo)
    ax.set_xticks(range(facies)); ax.set_xticklabels(SHORT, rotation=50, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(sites)); ax.set_yticklabels(LETTERS[:sites])
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    facts = {}
    fig = plt.figure(figsize=(W, 2.75))
    block_w, gap, left = 1.4, 0.1, 0.32
    top = 0.33
    for b, n in enumerate((3, 5)):
        B, pis = load(n)
        centered = B - B.mean(axis=0, keepdims=True)
        absent = pis <= ABSENT
        x0 = left + b * (2 * block_w + gap + 0.3)
        h = n * CELL_H
        ax1 = inch_axes(fig, x0, top, block_w, h)
        ax2 = inch_axes(fig, x0 + block_w + gap, top, block_w, h)
        vmax = np.abs(centered).max()
        heat(ax1, centered, column_ranks(centered), absent, vmax=vmax)
        heat(ax2, pis, column_ranks(pis), absent, tint=True)
        ax2.set_yticks([])
        ax1.text(0, 1.03, f"({'ab'[b]}) $N={n}$: bias change", transform=ax1.transAxes, ha="left", va="baseline")
        ax2.text(0, 1.03, "training share", transform=ax2.transAxes, ha="left", va="baseline")
        ax1.text(-0.04, 1.03, "Site", transform=ax1.transAxes, ha="right", va="baseline")
        facts[f"n{n}"] = {"absent_pairs": [[LETTERS[i], style.F3_NAMES[k]] for i, k in zip(*np.where(absent))],
                          "bias_max_abs_centered": float(vmax)}

    pooled = {}
    jobs = []
    for job in sorted(p for p in RUN_ROOT.iterdir() if p.is_dir()):
        n = 3 if "_n3_" in job.name else 5
        bias_w, row_w = job_pooled_within(job)
        jobs.append({"job": job.name, "bias": bias_w, "row": row_w})
        for seed in sorted(job.glob("seed_*")):
            hist, _ = load_labels(seed)
            pis = np.stack([hist[c] for c in sorted(hist)])
            _, biases = outc_channels(seed)
            B = np.stack([biases[c] for c in sorted(biases)])
            pooled.setdefault(n, []).append((B.ravel(), (pis > ABSENT).astype(int).ravel(), seed.parent.name))
    for n, runs in pooled.items():
        s = np.concatenate([r[0] for r in runs]); t = np.concatenate([r[1] for r in runs])
        facts[f"pooled_n{n}"] = {"n_runs": len(runs), "n_jobs": len({r[2] for r in runs}), "pairs": int(t.size),
                                 "negatives": int((t == 0).sum()), "sign_balanced_acc": balanced_accuracy(s > 0, t),
                                 "tpr": float((s[t == 1] > 0).mean()), "tnr": float((s[t == 0] <= 0).mean()),
                                 "base_rate": float(t.mean())}
    facts["per_job_within"] = jobs
    fig.text(0.5, 0.012, "Number: rank within facies column (1 = highest). Red/blue: above/below the column mean. Dash and outline: facies absent.",
             ha="center", va="bottom", fontsize=8)
    save(fig, "fig6_3_head_xray")
    write_facts("fig6_3", facts)
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
