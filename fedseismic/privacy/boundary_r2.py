"""Six-panel class-mix figure written to results/figures/boundary_r2.png.

Run from the repo root:

    python -m fedseismic.privacy.boundary_r2

Y is mean local holdout accuracy. X is class-mix R². The top row is the
last-upload softmax. The bottom row is Whispers. Secure aggregation and
server-side DP are omitted until a run saves the right quantities.
"""

from pathlib import Path
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.transforms import ScaledTranslation

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "figures" / "boundary_r2.png"
OUTPUT_COPY = ROOT / "results" / "figures" / "boundary_r2_methods.png"
OUTPUT_NODP = ROOT / "results" / "figures" / "boundary_r2_nodp.png"
LEGEND_ORDER = (
    "FedRoD",
    "FedBABU",
    "FedLC",
    "FedPer",
    "FedKPer λ=10, logit",
    "FedAvg + fine-tune",
    "FedKPer λ=10",
    "FedAvg",
    "FedProx",
    "Ditto",
    "FedKPer λ=0",
    "FedKPer λ=1",
)
DROPPED = {"SecAgg", "DP clip only", "DP σ=0.001", "DP σ=0.01"}

STYLE = {
    "FedAvg": ("#222222", "o"),
    "FedProx": ("#969696", "o"),
    "Ditto": ("#525252", "o"),
    "FedKPer λ=0": ("#9ecae1", "o"),
    "FedKPer λ=1": ("#6baed6", "o"),
    "FedKPer λ=10": ("#08519c", "o"),
    "FedKPer λ=10, logit": ("#08306b", "s"),
    "FedPer": ("#d94801", "D"),
    "FedLC": ("#222222", "s"),
    "FedAvg + fine-tune": ("#222222", "^"),
    "FedBABU": ("#7a0177", "p"),
    "FedRoD": ("#1b9e77", "h"),
    "SecAgg": ("#238b45", "*"),
    "DP clip only": ("#6a51a3", "x"),
    "DP σ=0.001": ("#cb181d", "x"),
    "DP σ=0.01": ("#fd8d3c", "x"),
}

SOFTMAX = {
    "BloodMNIST": {
        "FedAvg": (0.652, 0.882),
        "FedProx": (0.651, 0.884),
        "Ditto": (0.657, 0.897),
        "FedKPer λ=0": (0.598, 0.902),
        "FedKPer λ=1": (0.360, 0.920),
        "FedKPer λ=10": (0.330, 0.916),
        "FedKPer λ=10, logit": (-0.009, 0.935),
        "FedPer": (0.007, 0.929),
        "FedLC": (0.004, 0.930),
        "FedAvg + fine-tune": (0.652, 0.906),
    },
    "OrganCMNIST": {
        "FedAvg": (0.560, 0.928),
        "FedProx": (0.562, 0.933),
        "Ditto": (0.553, 0.920),
        "FedKPer λ=0": (0.559, 0.928),
        "FedKPer λ=1": (0.291, 0.941),
        "FedKPer λ=10": (0.263, 0.935),
        "FedKPer λ=10, logit": (-0.014, 0.937),
        "FedPer": (0.045, 0.921),
        "FedLC": (0.079, 0.937),
        "FedAvg + fine-tune": (0.560, 0.948),
    },
    "CIFAR-10": {
        "FedAvg": (0.813, 0.871),
        "FedKPer λ=10": (0.523, 0.879),
        "FedPer": (0.102, 0.882),
        "FedLC": (0.111, 0.886),
    },
}

NEW_RUNS = {
    ("BloodMNIST", "FedBABU"): "results/runs/fedbabu_blood/fedbabu_noniid_a0.1_lNA_dna",
    ("OrganCMNIST", "FedBABU"): "results/runs/fedbabu_organc/fedbabu_noniid_a0.1_lNA_dna",
    ("CIFAR-10", "FedBABU"): "results/runs/fedbabu_cifar/fedbabu_noniid_a0.1_lNA_dna",
    ("BloodMNIST", "FedRoD"): "results/runs/fedrod_blood/fedrod_noniid_a0.1_lNA_dna",
    ("OrganCMNIST", "FedRoD"): "results/runs/fedrod_organc/fedrod_noniid_a0.1_lNA_dna",
    ("CIFAR-10", "FedRoD"): "results/runs/fedrod_cifar/fedrod_noniid_a0.1_lNA_dna",
}
SEEDS = ("seed_0", "seed_1", "seed_42")

LOCAL_ROOTS = {
    ("BloodMNIST", "FedAvg"): "results/runs/whispers_blood/fedavg_noniid_a0.1_lNA_dna",
    ("BloodMNIST", "FedKPer λ=10"): "results/runs/whispers_blood/fedkper_noniid_a0.1_l10_dinfer",
    ("BloodMNIST", "FedPer"): "results/runs/whispers_blood/fedper_noniid_a0.1_lNA_dna",
    ("BloodMNIST", "FedLC"): "results/runs/whispers_blood_fedlc/fedavg_noniid_a0.1_lNA_dna",
    ("OrganCMNIST", "FedAvg"): "results/runs/whispers_organc/fedavg_noniid_a0.1_lNA_dna",
    ("OrganCMNIST", "FedKPer λ=10"): "results/runs/whispers_organc/fedkper_noniid_a0.1_l10_dinfer",
    ("OrganCMNIST", "FedPer"): "results/runs/whispers_organc/fedper_noniid_a0.1_lNA_dna",
    ("OrganCMNIST", "FedLC"): "results/runs/whispers_organc_fedlc/fedavg_noniid_a0.1_lNA_dna",
}


def _last_softmax_r2(run):
    """Class-mix R² of the last upload softmax in one saved seed."""
    import numpy as np

    from fedseismic.privacy.distribution import histogram_r2
    from fedseismic.privacy.log import load_labels
    from fedseismic.privacy.metrics import to_simplex

    histograms, _ = load_labels(run)
    last = {}
    with (Path(run) / "round_scores.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            client = int(record["client"])
            round_index = int(record["round"])
            previous = last.get(client)
            if previous is None or round_index >= previous[0]:
                last[client] = (round_index, np.asarray(record["local_softmax"], dtype=np.float64))
    if not last:
        raise RuntimeError(f"no round scores in {run}")
    clients = sorted(last)
    truths = [to_simplex(histograms[client]) for client in clients]
    hats = [last[client][1] for client in clients]
    return histogram_r2(truths, hats)


def _load_new_methods(softmax, whispers):
    """Read FedBABU and FedRoD from the run directories, not from a pasted table.

    Local accuracy and the softmax R² are computed from ``results.json`` and
    ``round_scores.jsonl``. Whispers R² comes from the scoring table only after
    that table's local accuracy matches the same seed's ``results.json``.
    """
    scored = {
        "FedBABU": pd.read_csv(ROOT / "results" / "tables" / "whispers_fedbabu.csv"),
        "FedRoD": pd.read_csv(ROOT / "results" / "tables" / "whispers_fedrod.csv"),
    }
    for (dataset, method), relative in NEW_RUNS.items():
        root = ROOT / relative
        if not root.is_dir():
            raise FileNotFoundError(root)
        softmax_values = []
        whisper_values = []
        local_values = []
        for seed in SEEDS:
            run = root / seed
            results_path = run / "results.json"
            if not results_path.is_file():
                raise FileNotFoundError(results_path)
            local = float(json.loads(results_path.read_text())["local_mean"])
            softmax_r2 = _last_softmax_r2(run)
            match = scored[method][
                (scored[method]["dataset"] == dataset) & (scored[method]["seed"] == seed)
            ]
            if len(match) != 1:
                raise RuntimeError(f"expected one Whispers row for {method} {dataset} {seed}")
            recorded = float(match["local_mean"].iloc[0])
            if abs(recorded - local) > 1e-6:
                raise RuntimeError(
                    f"{method} {dataset} {seed} scoring table local {recorded} "
                    f"does not match {results_path} ({local})"
                )
            whisper_r2 = float(match["whispers_r2"].iloc[0])
            softmax_values.append(softmax_r2)
            whisper_values.append(whisper_r2)
            local_values.append(local)
            print(
                f"loaded {method} {dataset} {seed} from {results_path} "
                f"softmax_r2 {softmax_r2:+.3f} whispers_r2 {whisper_r2:+.3f} local {local:.3f}"
            )
        point_y = sum(local_values) / len(local_values)
        softmax[dataset][method] = (sum(softmax_values) / len(softmax_values), point_y)
        whispers[dataset][method] = (sum(whisper_values) / len(whisper_values), point_y)


def _mean_local(root):
    values = []
    for seed in ("seed_0", "seed_1", "seed_42"):
        payload = json.loads((ROOT / root / seed / "results.json").read_text())
        values.append(float(payload["local_mean"]))
    return sum(values) / len(values)


def _tables():
    names = [
        "whispers_blood.csv",
        "whispers_organc.csv",
        "whispers_cifar.csv",
        "whispers_logit.csv",
        "whispers_dp.csv",
        "whispers_cifar_more.csv",
        "whispers_fedbabu.csv",
        "whispers_fedrod.csv",
    ]
    frames = [pd.read_csv(ROOT / "results" / "tables" / name) for name in names]
    frames[0]["dataset"] = "BloodMNIST"
    frames[1]["dataset"] = "OrganCMNIST"
    rows = pd.concat(frames, ignore_index=True)
    whispers = {name: {} for name in SOFTMAX}
    for (dataset, method), part in rows.groupby(["dataset", "method"]):
        if dataset not in whispers or method in DROPPED:
            continue
        x = float(part["whispers_r2"].mean())
        if "local_mean" in part.columns and part["local_mean"].notna().all():
            y = float(part["local_mean"].mean())
        else:
            y = _mean_local(LOCAL_ROOTS[(dataset, method)])
        whispers[dataset][method] = (x, y)
    softmax = {dataset: dict(points) for dataset, points in SOFTMAX.items()}
    more = pd.read_csv(ROOT / "results" / "tables" / "whispers_cifar_more.csv")
    for method, part in more.groupby("method"):
        if method in DROPPED:
            continue
        softmax["CIFAR-10"][method] = (
            float(part["softmax_r2"].mean()), float(part["local_mean"].mean()),
        )
    _load_new_methods(softmax, whispers)
    fine = pd.read_csv(ROOT / "results" / "tables" / "fedavg_finetune.csv")
    for dataset, key in (
        ("bloodmnist", "BloodMNIST"),
        ("organcmnist", "OrganCMNIST"),
        ("cifar10", "CIFAR-10"),
    ):
        y = float(fine.loc[fine["dataset"] == dataset, "finetune_local"].mean())
        whispers[key]["FedAvg + fine-tune"] = (whispers[key]["FedAvg"][0], y)
        if key == "CIFAR-10":
            softmax[key]["FedAvg + fine-tune"] = (softmax[key]["FedAvg"][0], y)
    return softmax, whispers


def _pixel_clusters(ax, items, radius=14):
    """Group methods whose markers would land on the same pixel."""
    remaining = list(items)
    clusters = []
    while remaining:
        seed = remaining.pop(0)
        _, x, y = seed
        sx, sy = ax.transData.transform((x, y))
        group = [seed]
        still = []
        for other in remaining:
            ox, oy = ax.transData.transform((other[1], other[2]))
            if (ox - sx) ** 2 + (oy - sy) ** 2 <= radius ** 2:
                group.append(other)
            else:
                still.append(other)
        remaining = still
        clusters.append(group)
    return clusters


def _draw_panel(fig, ax, table):
    order = {name: index for index, name in enumerate(STYLE)}
    items = [(name, x, y) for name, (x, y) in table.items()]
    missing = [name for name in ("FedBABU", "FedRoD") if name not in table]
    if missing:
        raise RuntimeError(f"missing {missing}")
    for group in _pixel_clusters(ax, items):
        group.sort(key=lambda item: order[item[0]])
        shifts = [(index - (len(group) - 1) / 2) * 13 for index in range(len(group))]
        if len(group) > 1:
            print("stacked " + ", ".join(name for name, _, _ in group))
        for (name, x, y), shift in zip(group, shifts):
            color, marker = STYLE[name]
            size = 70 if name == "FedRoD" else (48 if marker == "*" else 32)
            transform = ax.transData
            if len(group) > 1:
                ax.scatter([x], [y], c="#222222", s=8, zorder=2, linewidths=0)
                ax.annotate(
                    "", xy=(x, y), xytext=(0, shift), textcoords="offset points",
                    arrowprops={"arrowstyle": "-", "color": "#bdbdbd", "lw": 0.5},
                    zorder=2,
                )
                transform = ax.transData + ScaledTranslation(0, shift / 72, fig.dpi_scale_trans)
            ax.scatter(
                [x], [y], c=color, marker=marker, s=size, transform=transform,
                linewidths=0.6 if marker in {"x", "h"} else 0,
                edgecolors=color, zorder=4,
            )
            if name == "FedRoD":
                ax.annotate(
                    name, xy=(x, y), xytext=(7, shift + 4), textcoords="offset points",
                    fontsize=7, color=color, zorder=5,
                )
    return len(items)


def draw(softmax, whispers):
    ylim = {
        "BloodMNIST": (0.745, 0.955),
        "OrganCMNIST": (0.825, 0.965),
        "CIFAR-10": (0.705, 0.915),
    }
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.4), sharex=True)
    counts = []
    for col, dataset in enumerate(("BloodMNIST", "OrganCMNIST", "CIFAR-10")):
        for row, table in enumerate((softmax, whispers)):
            ax = axes[row, col]
            ax.set_xlim(-0.80, 0.95)
            ax.set_ylim(*ylim[dataset])
            count = _draw_panel(fig, ax, table[dataset])
            counts.append((dataset, row, count, sorted(table[dataset])))
            ax.axvline(0.0, color="#d9d9d9", linewidth=0.6, zorder=0)
            if row == 0:
                ax.set_title(dataset)
            if col == 0:
                ax.set_ylabel("Mean local accuracy")
            if row == 1:
                ax.set_xlabel("Class-mix R²")
    handles = []
    for name in LEGEND_ORDER:
        color, marker = STYLE[name]
        handles.append(plt.Line2D(
            [0], [0], marker=marker, color="none", markerfacecolor=color,
            markeredgecolor=color, markersize=9 if marker == "*" else 7, label=name,
        ))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(
        "Dirichlet α=0.1, mean of 3 seeds. Top row is the softmax attack. Bottom row is Whispers. A tick is the true value when markers share it.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))
    return fig, counts


def main():
    softmax, whispers = _tables()
    fig, counts = draw(softmax, whispers)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name("boundary_r2_writing.png")
    fig.savefig(temporary, dpi=160, format="png", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    os.replace(temporary, OUTPUT)
    OUTPUT_COPY.write_bytes(OUTPUT.read_bytes())
    OUTPUT_NODP.write_bytes(OUTPUT.read_bytes())
    for dataset, row, count, names in counts:
        print(f"{dataset} row {row}: {count} methods: {', '.join(names)}")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
