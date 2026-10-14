"""Current-data view of the privacy–personalization boundary.

Leakage is measured softmax 1 − TV. Upload and oracle runs are omitted:
their leakage was assigned, not measured. Utility scatters use three points
per method. The personalization scatter uses mean local accuracy. Higher is better.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASETS = {
    "BloodMNIST": "results/privacy_boundary/summary.csv",
    "OrganCMNIST": "results/privacy_boundary_organc/summary.csv",
}
UTILITY = (
    ("local_mean", "o", "local accuracy"),
    ("local_worst", "s", "worst client"),
    ("global_score", "^", "global accuracy"),
)
COLORS = {
    "fedavg": "#222222",
    "fedprox": "#737373",
    "fedkper λ=0": "#9ecae1",
    "fedkper λ=1": "#3182bd",
    "fedkper λ=10": "#08519c",
}


def _hard_split(frame):
    return frame[(frame["split"] == "noniid") & (np.isclose(frame["partition_alpha"], 0.1))].copy()


def _label(row):
    if row["algorithm"] != "fedkper":
        return row["algorithm"]
    return f"fedkper λ={row['lambda_cap']:g}"


def summarize(csv_path):
    frame = _hard_split(pd.read_csv(csv_path))
    dropped = (frame["algorithm"] == "fedbn") | (
        (frame["algorithm"] == "fedkper") & frame["fedkper_diversity"].isin(["upload", "oracle"])
    )
    frame = frame.loc[~dropped].copy()
    frame["label"] = frame.apply(_label, axis=1)
    grouped = frame.groupby(
        ["algorithm", "lambda_cap", "fedkper_diversity", "label"], dropna=False,
    )
    rows = []
    for keys, part in grouped:
        row = {name: keys[index] for index, name in enumerate(
            ["algorithm", "lambda_cap", "fedkper_diversity", "label"]
        )}
        for metric in (
            "leakage", "local_mean", "local_worst",
            "global_score", "personalization_gap",
        ):
            row[metric] = float(part[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _points(csv_path):
    return summarize(csv_path).reset_index(drop=True)


def _color(label):
    return COLORS.get(label, "#444444")


def plot_utilities(output_path, x_key="leakage", x_label="Leakage (1 − TV)"):
    """One panel per dataset. Each method is three points at the same leakage."""
    output_path = Path(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=True)
    metric_handles = []
    method_handles = []
    seen_methods = set()
    for ax, (dataset, csv_path) in zip(axes, DATASETS.items()):
        summary = _points(csv_path)
        for metric, marker, name in UTILITY:
            handle = ax.scatter(
                [], [], marker=marker, c="#444444", s=36, label=name,
            )
            if len(metric_handles) < len(UTILITY):
                metric_handles.append(handle)
            for _, point in summary.iterrows():
                method = ax.scatter(
                    point[x_key], point[metric],
                    marker=marker, c=_color(point["label"]), s=42, linewidths=0,
                )
                if point["label"] not in seen_methods:
                    method.set_label(point["label"])
                    method_handles.append(method)
                    seen_methods.add(point["label"])
        ax.set_title(dataset)
        ax.set_xlabel(x_label)
        ax.set_xlim(0.35, 0.78)
        ax.grid(True, linewidth=0.4, alpha=0.45)
    axes[0].set_ylabel("Accuracy")
    fig.legend(
        metric_handles + method_handles,
        [h.get_label() for h in metric_handles + method_handles],
        loc="lower center", ncol=4, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle("Non-IID α=0.1, seed average. FedBN matches FedAvg and is omitted.", fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_personalization(output_path, x_key="leakage", x_label="Leakage (1 − TV)"):
    """One point per method. Y is mean local accuracy, so higher is better."""
    output_path = Path(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    handles = []
    seen = set()
    for ax, (dataset, csv_path) in zip(axes, DATASETS.items()):
        summary = _points(csv_path)
        for _, point in summary.iterrows():
            handle = ax.scatter(
                point[x_key], point["local_mean"],
                c=_color(point["label"]), s=46, linewidths=0,
            )
            if point["label"] not in seen:
                handle.set_label(point["label"])
                handles.append(handle)
                seen.add(point["label"])
        ax.set_title(dataset)
        ax.set_xlabel(x_label)
        ax.set_xlim(0.35, 0.78)
        ax.grid(True, linewidth=0.4, alpha=0.45)
    axes[0].set_ylabel("Personalization\n(mean local accuracy)")
    fig.legend(
        handles, [h.get_label() for h in handles],
        loc="lower center", ncol=3, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, -0.1),
    )
    fig.suptitle("Non-IID α=0.1, seed average. Higher is better personalization.", fontsize=11)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_advantage(frame, output_path):
    """One point per method. X is the strongest attack's advantage over its baseline."""
    output_path = Path(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    handles = []
    seen = set()
    for ax, dataset in zip(axes, ("BloodMNIST", "OrganCMNIST")):
        part = frame[frame["dataset"] == dataset]
        for _, point in part.iterrows():
            handle = ax.scatter(
                point["worst_advantage"], point["local_mean"],
                c=_color(point["label"]), s=46, linewidths=0,
            )
            if point["label"] not in seen:
                handle.set_label(point["label"])
                handles.append(handle)
                seen.add(point["label"])
        ax.set_title(dataset)
        ax.set_xlabel("Worst-case leakage advantage")
        ax.grid(True, linewidth=0.4, alpha=0.45)
    axes[0].set_ylabel("Personalization\n(mean local accuracy)")
    fig.legend(
        handles, [h.get_label() for h in handles],
        loc="lower center", ncol=3, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, -0.1),
    )
    fig.suptitle(
        "Non-IID α=0.1, seed average. Higher is better personalization.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path
