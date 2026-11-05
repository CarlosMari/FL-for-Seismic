"""Fig 6.3 (replaces the rank heatmap): what the segmentation head reveals.

All eight seismic_f3 jobs (FedAvg/FedBN x batch/group norm x N=3,5), seeds
0, 1, 42, round-20 head update (head_outc/). One point per site-facies pair.
(a) Bias change, divided by the job's standard deviation over its pairs
    (sign kept; one constant per job, so correlations are unchanged), by facies; absent pairs as crosses. Sign rule: present iff > 0.
(b) Within-facies Pearson between log10 pixel count and bias change, per
    facies and job (8 dots per facies). tab:s-head is the mean over facies.
Writes figures/ch6/fig6_3_head.{pdf,png} and data/fig6_3_head.json.
"""
from thesis_ch6_common import *

from fedseismic.privacy.seismic_channel import ABSENT, outc_channels, site_counts, _within_pearson
from fedseismic.privacy.log import load_labels

RUN_ROOT = ROOT / "results/runs/seismic_f3"
SHORT = ["U. North Sea", "M. North Sea", "L. North Sea", "Rijnland/Chalk", "Scruff", "Zechstein"]
COLORS = style.FACIES_COLORS[:6] if hasattr(style, "FACIES_COLORS") else plt.cm.tab10.colors[:6]


def collect():
    pairs = []          # (job, facies, scaled bias, present)
    within = []         # (facies, centered logcount, centered scaled bias)
    per_job = {}
    pis_job = []
    for job in sorted(RUN_ROOT.glob("*")):
        logc_all, b_all = [], []
        for seed in sorted(job.glob("seed_*")):
            counts = site_counts(seed)
            hist, _ = load_labels(seed)
            _, biases = outc_channels(seed)
            clients = sorted(biases)
            B = np.stack([biases[c] for c in clients])
            pis = np.stack([np.asarray(hist[c], float) / np.sum(hist[c]) for c in clients])
            pis_job.append(pis)
            logc_all.append(np.log10(np.maximum(counts[clients], 1.0)))
            b_all.append(B)
        L, Bj = np.concatenate(logc_all), np.concatenate(b_all)
        Bj = Bj / Bj.std()
        P = np.concatenate(pis_job); pis_job.clear()
        for i in range(Bj.shape[0]):
            for k in range(Bj.shape[1]):
                pairs.append((job.name, k, float(Bj[i, k]), bool(P[i, k] > ABSENT)))
        per_job[job.name] = _within_pearson(L, Bj)
        from scipy.stats import pearsonr
        for k in range(L.shape[1]):
            if np.std(L[:, k]) > 1e-12 and np.std(Bj[:, k]) > 1e-12:
                within.append((k, job.name, float(pearsonr(L[:, k], Bj[:, k]).statistic)))
    return pairs, within, per_job


def main():
    pairs, within, per_job = collect()
    present = np.array([p[3] for p in pairs]); vals = np.array([p[2] for p in pairs]); fac = np.array([p[1] for p in pairs])
    facts = {
        "n_pairs": int(len(pairs)), "n_absent": int((~present).sum()),
        "absent_below_zero": int((vals[~present] < 0).sum()),
        "present_above_zero_frac": float((vals[present] > 0).mean()),
        "within_pearson_per_job": per_job,
        "present_above_zero_by_facies": {SHORT[k]: [int(((fac == k) & present & (vals > 0)).sum()), int(((fac == k) & present).sum())] for k in range(6)},
        "absent_by_facies": {SHORT[k]: int(((fac == k) & ~present).sum()) for k in range(6)},
    }
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(W, 2.6))
    ax = inch_axes(fig, 0.55, 0.12, 3.05, 1.85)
    ax.axhline(0, color="#777777", lw=0.6, ls="--", zorder=1)
    for k in range(6):
        m = (fac == k) & present
        ax.scatter(k + rng.uniform(-0.28, 0.28, m.sum()), vals[m], s=6, color=COLORS[k], alpha=0.55, lw=0, zorder=2)
        m = (fac == k) & ~present
        ax.scatter(k + rng.uniform(-0.28, 0.28, m.sum()), vals[m], s=16, marker="x", color="k", lw=0.8, zorder=3)
    ax.set_xticks(range(6)); ax.set_xticklabels(SHORT, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Bias change (scaled per job)")
    ax.text(0.02, 0.97, "(a) Presence", transform=ax.transAxes, va="top", fontsize=9)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#888888", ms=3, label="facies present"),
                       Line2D([], [], marker="x", ls="", color="k", ms=4, label="facies absent")],
              loc="lower left", frameon=False, handletextpad=0.2, borderaxespad=0.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax = inch_axes(fig, 4.15, 0.12, 2.25, 1.85)
    ks = np.array([w[0] for w in within]); rs = np.array([w[2] for w in within])
    ax.axhline(0, color="#777777", lw=0.6, ls="--", zorder=1)
    for k in range(6):
        m = ks == k
        ax.scatter(k + rng.uniform(-0.18, 0.18, m.sum()), rs[m], s=12, color=COLORS[k], lw=0.3, edgecolor="k", zorder=2)
    facts["per_facies_r"] = [[SHORT[k], j, r] for k, j, r in within]
    facts["positive_r"] = [int((rs > 0).sum()), int(len(rs))]
    ax.set_xticks(range(6)); ax.set_xticklabels(SHORT, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Pearson $r$ across sites")
    ax.text(0.02, 0.03, "(b) Amount", transform=ax.transAxes, va="bottom", fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    write_facts("fig6_3_head", facts)
    save(fig, "fig6_3_head")
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
