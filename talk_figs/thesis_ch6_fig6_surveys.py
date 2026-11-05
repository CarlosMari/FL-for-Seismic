"""Fig 6.6: three-survey arms per survey and per class, from thesis_ch6_rescore_k3.py."""
from thesis_ch6_common import *

import pandas as pd

ARMS = [("solo", "Solo", "#6A3D9A", None), ("fedpersurvey", "Per-survey heads", "#2A9D8F", None),
        ("fedhca2", "FedHCA2-style (control)", "white", "/////")]
SURVEYS = ["F3", "Parihaka", "Penobscot"]
NAMES = {"F3": style.F3_NAMES, "Parihaka": style.PARIHAKA_NAMES, "Penobscot": style.PENOBSCOT_NAMES}
CONTROL_EDGE = "#B5651D"


def bar(ax, x, value, width, spec):
    _, _, color, hatch = spec
    ax.bar(x, value, width, color=color, hatch=hatch, ec=CONTROL_EDGE if hatch else "white", lw=0.5)


def tidy(ax):
    ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)


def main():
    summary = pd.read_csv(DATA / "k3_summary.csv")
    per = pd.read_csv(DATA / "k3_per_class.csv")
    fig = plt.figure(figsize=(W, 4.5))
    ax = inch_axes(fig, 0.46, 0.32, W - 0.5, 1.35)
    width = 0.26
    for s, survey in enumerate(SURVEYS):
        for a, spec in enumerate(ARMS):
            row = summary[(summary.arm == spec[0]) & (summary.survey == survey)]
            x = s + (a - 1) * width
            if row.empty:
                ax.text(x, 0.02, "no solo\nrun", ha="center", va="bottom", fontsize=8)
                continue
            value = float(row.miou_present.iloc[0])
            bar(ax, x, value, width * 0.94, spec)
            ax.text(x, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    tidy(ax)
    ax.set_xticks(range(3)); ax.set_xticklabels(SURVEYS, fontsize=9)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("Present-class mIoU", labelpad=2)
    fig.text(0.008, 1 - 0.3 / 4.5, "(a)", ha="left", va="top")

    ax = inch_axes(fig, 0.46, 2.16, W - 0.5, 1.35)
    width = 0.28
    x0, ticks, labels = 0.0, [], []
    for survey in SURVEYS:
        classes = NAMES[survey]
        for k, name in enumerate(classes):
            for a, spec in enumerate(ARMS):
                row = per[(per.arm == spec[0]) & (per.survey == survey) & (per["class"] == k)]
                if not row.empty:
                    bar(ax, x0 + k + (a - 1) * width, float(row.iou.iloc[0]), width * 0.94, spec)
            ticks.append(x0 + k); labels.append(name)
        ax.text(x0 + (len(classes) - 1) / 2, 1.03, survey + ("" if not summary[(summary.arm == "solo") & (summary.survey == survey)].empty else " (no solo run)"),
                ha="center", va="bottom", fontsize=8)
        if survey != "Penobscot":
            ax.axvline(x0 + len(classes) - 0.25, color="#999999", lw=0.4, ls=(0, (2, 2)))
        x0 += len(classes) + 0.5
    tidy(ax)
    ax.set_xlim(-0.6, x0 - 0.9)
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("IoU", labelpad=2)
    fig.text(0.008, 1 - 2.14 / 4.5, "(b)", ha="left", va="top")

    handles = [mpatches.Patch(fc=c, ec=CONTROL_EDGE if h else "none", hatch=h, lw=0.5, label=n) for _, n, c, h in ARMS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.5, handletextpad=0.4, columnspacing=1.5)
    save(fig, "fig6_6_three_surveys")


if __name__ == "__main__":
    main()
