"""Fig 6.1: F3 sites: strip map, one median-error crossline per site, facies mix.

Needs figures/ch6/data/fig6_2.json (crossline picks) from thesis_ch6_fig2_triptych.py.
"""
from thesis_ch6_common import *

from fedseismic.data.partition import partition_noniid, split_client_local_test
from fedseismic.privacy.log import load_labels

LABELS = ROOT / "datasets/seismic/data/train/train_labels.npy"
SEISMIC = ROOT / "datasets/seismic/data/train/train_seismic.npy"
RUN = str(ROOT / "results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n{}_batchnorm/seed_0")


def split_facts(n, labs):
    parts = partition_noniid(labs.shape[1], n)
    train, tail = split_client_local_test(parts, 0.2, rng=None)
    hist, _ = load_labels(RUN.format(n))
    rows = []
    for i in range(n):
        tr = labs[:, train[i], :]; te = labs[:, tail[i], :]
        rows.append({"site": LETTERS[i], "block": [int(parts[i][0]), int(parts[i][-1])],
                     "train": [int(train[i][0]), int(train[i][-1])], "holdout": [int(tail[i][0]), int(tail[i][-1])],
                     "pi_json_pct": [100 * float(v) for v in hist[i]],
                     "train_cube_pct": [100 * float((tr == k).mean()) for k in range(6)],
                     "zechstein_train_px": int((tr == 5).sum()), "zechstein_holdout_px": int((te == 5).sum())})
    return rows


def main():
    labs = np.load(LABELS); seis = np.load(SEISMIC)
    facts = {"n3": split_facts(3, labs), "n5": split_facts(5, labs)}
    picks = json.loads((DATA / "fig6_2.json").read_text())["rows"]
    rows = facts["n3"]

    fig = plt.figure(figsize=(W, 3.95))
    # (a) strip map
    ax = inch_axes(fig, 0.3, 0.2, W - 0.4, 0.3)
    for i, row in enumerate(rows):
        lo, hi = row["train"]; tlo, thi = row["holdout"]
        ax.barh(0, hi - lo + 1, left=lo, height=1, color=SITE_COLORS[i], ec="white", lw=0.5)
        ax.barh(0, thi - tlo + 1, left=tlo, height=1, fc="white", ec=SITE_COLORS[i], hatch="//////", lw=0.5)
        ax.text((lo + hi) / 2, 0, f"Site {row['site']} train", color="white", ha="center", va="center", fontsize=8)
        ax.text((tlo + thi) / 2, 0.52, f"{tlo}-{thi}", ha="center", va="bottom", fontsize=8)
    ax.set_xlim(0, 701); ax.set_ylim(-0.5, 0.5); ax.set_yticks([])
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 700])
    ax.tick_params(pad=1)
    ax.set_xlabel("Crossline index (hatched: 20% holdout tail)", labelpad=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.008, 1 - 0.2 / 3.95, "(a)", ha="left", va="top")

    # (b) one median-error crossline per site, true aspect
    sec_w = (W - 0.3 - 0.1 - 2 * 0.06) / 3
    sec_h = sec_w * 255 / 401
    y = 0.95
    fig.text(0.008, 1 - y / 3.95, "(b)", ha="left", va="top")
    for i in range(3):
        xl = picks[i]["crossline"]
        ax = inch_axes(fig, 0.3 + i * (sec_w + 0.06), y, sec_w, sec_h)
        facies_over(ax, seis[:, xl, :].T, labs[:, xl, :].T)
        panel_label(ax, f"Site {LETTERS[i]}, crossline {xl}")
        facts["n3"][i]["section_crossline"] = xl

    # (c) facies mix on training pixels
    y2 = y + sec_h + 0.12
    fig.text(0.008, 1 - y2 / 3.95, "(c)", ha="left", va="top")
    ax = inch_axes(fig, 0.56, y2, W - 0.56 - 1.72, 0.84)
    for i, row in enumerate(rows):
        left = 0.0
        for k, pct in enumerate(row["pi_json_pct"]):
            if pct <= 0:
                continue
            ax.barh(i, pct, left=left, height=0.72, color=style.FACIES_COLORS[k], ec="white", lw=0.5)
            if pct >= 5 and k < 4:
                ax.text(left + pct / 2, i, f"{pct:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if k in (0, 2) else "black")
            left += pct
    ax.set_xlim(0, 100); ax.set_ylim(2.5, -0.5)
    ax.set_yticks(range(3)); ax.set_yticklabels([f"Site {c}" for c in "ABC"])
    ax.tick_params(axis="y", length=0, pad=2); ax.tick_params(axis="x", pad=1)
    ax.set_xlabel("Share of training pixels (%)", labelpad=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    b_tail = rows[1]["holdout"]
    for i, row in enumerate(rows):
        pct = row["pi_json_pct"]
        note = f"Scruff {pct[4]:.1f}, Zechstein {pct[5]:.1f}"
        if i == 1:
            note += f"\nZechstein only in holdout, {b_tail[0]}-{b_tail[1]}"
        ax.text(101, i, note, ha="left", va="center", fontsize=8, linespacing=0.95, clip_on=False)
    fig.legend(handles=facies_handles(style.F3_NAMES), loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.1, handletextpad=0.3, columnspacing=0.9)
    save(fig, "fig6_1_f3_sites")
    write_facts("fig6_1", facts)
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
