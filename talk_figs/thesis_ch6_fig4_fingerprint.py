"""Fig 6.4: static PCA of centered BN means (BN) and delta_proj sketches (GN)."""
from thesis_ch6_common import *

from sklearn.decomposition import PCA

from f5_fingerprint import HDD, SEEDS, centered_features, load_view_obs, nn_top1_and_margin, pick_seed

JOB = "fedavg_noniid_a0.1_lNA_dna_n{}_{}"


def job_scores(n, norm, channel):
    per_seed, first = {}, {}
    for seed in SEEDS:
        obs = load_view_obs(HDD / JOB.format(n, norm) / seed, channel)
        keys, X, layers = centered_features(obs)
        top1, margin = nn_top1_and_margin(keys, X)
        per_seed[seed] = {"top1": top1, "margin": margin, "n_uploads": len(keys)}
        names = sorted(next(iter(next(iter(obs.values())).values())))
        keys_f, X_f, _ = centered_features(obs, lambda name: name == names[0])
        first[seed] = nn_top1_and_margin(keys_f, X_f)[0]
    pick, med_top, med_margin = pick_seed(per_seed)
    return {"per_seed": per_seed, "mean_top1": float(np.mean([v["top1"] for v in per_seed.values()])),
            "first_layer": first, "first_layer_mean": float(np.mean(list(first.values()))),
            "first_layer_name": names[0], "n_layers": len(layers), "picked": pick,
            "median_top1": med_top, "median_margin": med_margin}


def scatter(ax, n, norm, channel, seed):
    obs = load_view_obs(HDD / JOB.format(n, norm) / seed, channel)
    keys, X, _ = centered_features(obs)
    pca = PCA(n_components=2)
    xy = pca.fit_transform(X)
    rounds = np.array([k[0] for k in keys]); sites = np.array([k[1] for k in keys])
    for c in range(n):
        keep = sites == c
        ax.scatter(xy[keep, 0], xy[keep, 1], s=3 + 1.3 * rounds[keep],
                   c=SITE_COLORS[c], marker=SITE_MARKERS[c], alpha=0.75, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.0f}%)", labelpad=2)
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.0f}%)", labelpad=2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.margins(0.08)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.16 * (hi - lo), hi + 0.12 * (hi - lo))
    return [float(v) for v in pca.explained_variance_ratio_]


def main():
    facts = {"bn_n3": job_scores(3, "batchnorm", "bn"), "bn_n5": job_scores(5, "batchnorm", "bn"),
             "gn_n3": job_scores(3, "groupnorm", "proj"), "gn_n5": job_scores(5, "groupnorm", "proj")}
    fig = plt.figure(figsize=(W, 3.1))
    panels = [("bn_n3", "batchnorm", "bn", "(a)", "Batch norm: BN running means"),
              ("gn_n3", "groupnorm", "proj", "(b)", "Group norm: update sketch")]
    for j, (key, norm, channel, tag, name) in enumerate(panels):
        ax = inch_axes(fig, 0.3 + j * 3.15, 0.1, 2.9, 2.45)
        seed = facts[key]["picked"]
        facts[key]["pca_var"] = scatter(ax, 3, norm, channel, seed)
        panel_label(ax, f"{tag} {name}")
        corner_note(ax, f"top-1 NN {facts[key]['per_seed'][seed]['top1']:.3f} ({seed.replace('_', ' ')})\n"
                        f"3-seed mean {facts[key]['mean_top1']:.3f}; chance 0.33")
    handles = [plt.Line2D([], [], ls="", marker=SITE_MARKERS[c], ms=5, mfc=SITE_COLORS[c], mec="none",
                          label=f"Site {LETTERS[c]}") for c in range(3)]
    handles += [plt.Line2D([], [], ls="", marker="o", ms=np.sqrt(3 + 1.3 * r), mfc="#888888", mec="none",
                           label=f"round {r}") for r in (1, 20)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.3, columnspacing=1.4)
    save(fig, "fig6_4_fingerprint")
    write_facts("fig6_4", facts)
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
