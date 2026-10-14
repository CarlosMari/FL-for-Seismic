"""Scatter + Pareto front of leakage vs personalization gap."""

from pathlib import Path

import numpy as np
import pandas as pd


def pareto_mask(leakage, gap):
    """Non-dominated points: lower leakage, higher personalization gap."""
    leakage = np.asarray(leakage, dtype=np.float64)
    gap = np.asarray(gap, dtype=np.float64)
    keep = np.zeros(len(leakage), dtype=bool)
    for index in range(len(leakage)):
        if not np.isfinite(leakage[index]) or not np.isfinite(gap[index]):
            continue
        dominated = False
        for other in range(len(leakage)):
            if index == other:
                continue
            if not np.isfinite(leakage[other]) or not np.isfinite(gap[other]):
                continue
            better_or_equal = leakage[other] <= leakage[index] and gap[other] >= gap[index]
            strictly_better = leakage[other] < leakage[index] or gap[other] > gap[index]
            if better_or_equal and strictly_better:
                dominated = True
                break
        keep[index] = not dominated
    return keep


def plot_boundary(csv_path, output_path=None, ax=None):
    frame = pd.read_csv(csv_path)
    if "leakage" not in frame or "personalization_gap" not in frame:
        raise ValueError("summary CSV needs leakage and personalization_gap columns")
    grouped = frame.groupby(
        ["algorithm", "split", "partition_alpha", "lambda_cap", "fedkper_diversity"],
        dropna=False,
    ).agg({"leakage": "mean", "personalization_gap": "mean"}).reset_index()
    mask = pareto_mask(grouped["leakage"], grouped["personalization_gap"])
    import matplotlib.pyplot as plt

    owned = ax is None
    if owned:
        _, ax = plt.subplots(figsize=(7, 5))
    for algorithm, subset in grouped.groupby("algorithm"):
        ax.scatter(
            subset["leakage"], subset["personalization_gap"],
            label=algorithm, s=48,
        )
    front = grouped.loc[mask].sort_values("leakage")
    if not front.empty:
        ax.plot(front["leakage"], front["personalization_gap"], linestyle="--", color="black",
                label="pareto")
    ax.set_xlabel("Leakage (1 - TV)")
    ax.set_ylabel("Personalization gap")
    ax.legend()
    ax.set_title("Privacy–personalization boundary")
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(output_path, bbox_inches="tight")
    return grouped, mask
