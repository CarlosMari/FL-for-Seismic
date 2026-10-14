"""Decaf on saved classifier deltas.

A class is marked absent when no weight in its last-layer row moves up.
The continuous score is the sum of the positive entries in that row.
Proportions for the classes that remain are a line from that score to the
count, fit on shadow clients that use the victim's loss. FedPer has no
uploaded head, so it is not scored.
"""

from pathlib import Path
import json

import numpy as np
from sklearn.linear_model import LinearRegression

from fedseismic.privacy.head_channels import _client_id
from fedseismic.privacy.log import load_labels
from fedseismic.privacy.mechanism import (
    DISPLAY,
    _percentile,
    bootstrap_seed_mean,
    iter_runs,
    prior_presence_auc,
)
from fedseismic.privacy.metrics import to_simplex
from fedseismic.privacy.distribution import histogram_r2
from fedseismic.privacy.row_delta import _loo_r2, _pearson, _presence_auc

ABSENT = 1e-3


def positive_sum(weight_delta):
    """Sum of positive entries in each classifier row."""
    flat = np.asarray(weight_delta, dtype=np.float64).reshape(np.asarray(weight_delta).shape[0], -1)
    return np.clip(flat, 0.0, None).sum(axis=1)


def row_has_positive(weight_delta):
    """True when a row contains at least one strictly positive change."""
    flat = np.asarray(weight_delta, dtype=np.float64).reshape(np.asarray(weight_delta).shape[0], -1)
    return (flat > 0.0).any(axis=1)


def client_scores(run_dir):
    """One row per client. The hard rule fires only if every participation is non-positive."""
    histograms, _ = load_labels(run_dir)
    sums = {}
    positive = {}
    for path in sorted((Path(run_dir) / "head").glob("round_*_client_*.npz")):
        with np.load(path) as handle:
            weight = np.asarray(handle["weight_delta"], dtype=np.float64)
        client = _client_id(path)
        sums.setdefault(client, []).append(positive_sum(weight))
        positive.setdefault(client, []).append(row_has_positive(weight))
    if not sums:
        raise FileNotFoundError(f"no head deltas in {run_dir}")
    rows = []
    for client in sorted(sums):
        if client not in histograms:
            continue
        stacked = np.stack(positive[client])
        rows.append({
            "client": int(client),
            "pi": to_simplex(histograms[client]),
            "score": np.mean(np.stack(sums[client]), axis=0),
            "hard_present": stacked.any(axis=0),
        })
    return rows


def hard_accuracy(rows):
    """Accuracy of the hard rule over client-class pairs."""
    correct = 0
    total = 0
    for row in rows:
        present = row["pi"] > ABSENT
        correct += int(np.sum(row["hard_present"] == present))
        total += int(present.size)
    if total == 0:
        return float("nan")
    return correct / total


def majority_accuracy(rows):
    """Accuracy of calling every class present or every class absent, whichever fits better."""
    present = np.concatenate([row["pi"] > ABSENT for row in rows])
    rate = float(present.mean()) if len(present) else float("nan")
    return float(max(rate, 1.0 - rate))


def presence_auc(rows):
    return _presence_auc([row["pi"] for row in rows], [row["score"] for row in rows])


def _project(scores, present):
    """Zero the classes the hard rule calls absent, then put the rest on the simplex."""
    clipped = np.clip(np.asarray(scores, dtype=np.float64), 0.0, None)
    clipped = clipped * np.asarray(present, dtype=np.float64)
    total = float(clipped.sum())
    if total < 1e-18:
        return np.full(clipped.shape, 1.0 / clipped.size)
    return clipped / total


def shadow_proportion_r2(real_rows, shadow_scores, shadow_mixes):
    """Line from positive-sum to class share, fit on shadows, scored on real clients.

    Classes the hard rule calls absent are set to zero before the simplex projection.
    Shadows are not the victim clients.
    """
    shadow_scores = np.asarray(shadow_scores, dtype=np.float64)
    shadow_mixes = np.asarray(shadow_mixes, dtype=np.float64)
    present = shadow_mixes > ABSENT
    if int(present.sum()) < 4:
        return {"r2": float("nan"), "pearson": float("nan"), "loo_r2": float("nan")}
    model = LinearRegression().fit(shadow_scores[present].reshape(-1, 1), shadow_mixes[present])
    truths = []
    predicted = []
    raw = []
    for row in real_rows:
        estimate = model.predict(np.asarray(row["score"], dtype=np.float64).reshape(-1, 1))
        truths.append(row["pi"])
        predicted.append(_project(estimate, row["hard_present"]))
        raw.append(row["score"])
    return {
        "r2": histogram_r2(truths, predicted),
        "pearson": _pearson(truths, predicted),
        "loo_r2": _loo_r2(truths, raw) if len(truths) >= 2 else float("nan"),
    }


def score_saved(n_draws=1000):
    """Presence table for the prior-smoothing endpoints. Proportions need shadows."""
    grouped = {}
    for run in iter_runs():
        if abs(run["prior_smooth"]) > 1e-9 and abs(run["prior_smooth"] - 1.0) > 1e-9:
            continue
        rows = client_scores(run["path"])
        grouped.setdefault((run["dataset"], run["prior_smooth"]), []).append(rows)
    table = []
    for (dataset, smooth), groups in sorted(grouped.items()):
        method = "FedLC" if abs(smooth) < 1e-9 else "FedAvg"
        name = DISPLAY.get(dataset, dataset)
        auc, auc_low, auc_high, _ = bootstrap_seed_mean(groups, presence_auc, n_draws=n_draws)
        floor, _, _, _ = bootstrap_seed_mean(groups, prior_presence_auc, n_draws=n_draws)
        accuracy, acc_low, acc_high, _ = bootstrap_seed_mean(groups, hard_accuracy, n_draws=n_draws)
        base = float(np.mean([majority_accuracy(group) for group in groups]))
        table.append({
            "dataset": name,
            "method": method,
            "presence_auc": auc,
            "presence_auc_low": auc_low,
            "presence_auc_high": auc_high,
            "floor": floor,
            "leakage": auc - floor,
            "hard_accuracy": accuracy,
            "hard_accuracy_low": acc_low,
            "hard_accuracy_high": acc_high,
            "majority_accuracy": base,
            "proportion_r2": float("nan"),
            "proportion_pearson": float("nan"),
            "proportion_loo_r2": float("nan"),
        })
    return table


def _shadow_file(run_dir):
    relative = Path(run_dir).resolve().relative_to(Path("results/runs").resolve())
    return Path("results/attacks/shadows") / relative / "shadow.npz"


def _bootstrap_proportions(bundles, n_draws=1000):
    """Resample real clients inside each seed. The shadow line stays fixed.

    One draw scores R², Pearson, and leave-one-out R² together.
    """
    from fedseismic.privacy.mechanism import _percentile

    keys = ("r2", "pearson", "loo_r2")
    point = {key: [] for key in keys}
    for rows, positive, mixes in bundles:
        scored = shadow_proportion_r2(rows, positive, mixes)
        for key in keys:
            point[key].append(scored[key])
    point = {key: float(np.nanmean(values)) for key, values in point.items()}
    rng = np.random.RandomState(0)
    draws = {key: [] for key in keys}
    for _ in range(n_draws):
        scored = {key: [] for key in keys}
        for rows, positive, mixes in bundles:
            index = rng.randint(0, len(rows), size=len(rows))
            sample = shadow_proportion_r2([rows[item] for item in index], positive, mixes)
            for key in keys:
                scored[key].append(sample[key])
        for key in keys:
            draws[key].append(float(np.nanmean(scored[key])))
    out = {}
    for key in keys:
        low, high = _percentile(draws[key])
        out[key] = (point[key], low, high)
    return out


def proportion_table(n_draws=1000):
    """Fill proportion columns from shadow caches. Missing caches are left blank."""
    grouped = {}
    for run in iter_runs():
        if abs(run["prior_smooth"]) > 1e-9 and abs(run["prior_smooth"] - 1.0) > 1e-9:
            continue
        path = _shadow_file(run["path"])
        if not path.is_file():
            continue
        with np.load(path) as handle:
            if "positive" not in handle.files:
                continue
            positive = np.asarray(handle["positive"], dtype=np.float64)
            mixes = np.asarray(handle["mixes"], dtype=np.float64)
        grouped.setdefault((run["dataset"], run["prior_smooth"]), []).append(
            (client_scores(run["path"]), positive, mixes)
        )
    rows = []
    for (dataset, smooth), bundles in sorted(grouped.items()):
        method = "FedLC" if abs(smooth) < 1e-9 else "FedAvg"
        record = {"dataset": DISPLAY.get(dataset, dataset), "method": method}
        scored = _bootstrap_proportions(bundles, n_draws=n_draws)
        for key, column in (
            ("r2", "proportion_r2"),
            ("pearson", "proportion_pearson"),
            ("loo_r2", "proportion_loo_r2"),
        ):
            point, low, high = scored[key]
            record[column] = point
            record[f"{column}_low"] = low
            record[f"{column}_high"] = high
        rows.append(record)
        print(
            f"  {record['dataset']} {record['method']} R² {record['proportion_r2']:+.3f}",
            flush=True,
        )
    return rows


def fedavg_proportion_ok(rows):
    """Positive control: ordinary training's shadow-calibrated mix is above the floor."""
    checks = []
    for row in rows:
        if row["method"] != "FedAvg":
            continue
        checks.append(row["proportion_r2"] > 0.2 and row["proportion_pearson"] > 0.4)
    return bool(checks) and all(checks)


def fedavg_presence_ok(table):
    """Positive control: FedAvg's continuous score beats the population floor on every dataset."""
    checks = []
    for row in table:
        if row["method"] != "FedAvg":
            continue
        checks.append(row["presence_auc"] > row["floor"] and row["presence_auc"] > 0.7)
    return bool(checks) and all(checks)


def main():
    import pandas as pd

    table = score_saved()
    output = Path("results/tables/decaf.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(output, index=False)
    ok = fedavg_presence_ok(table)
    print("FedAvg presence control", "pass" if ok else "FAIL")
    for row in table:
        print(
            f"  {row['dataset']} {row['method']} auc {row['presence_auc']:.3f} "
            f"[{row['presence_auc_low']:.3f}, {row['presence_auc_high']:.3f}] "
            f"floor {row['floor']:.3f} leakage {row['leakage']:+.3f} "
            f"hard {row['hard_accuracy']:.3f} "
            f"[{row['hard_accuracy_low']:.3f}, {row['hard_accuracy_high']:.3f}] "
            f"majority {row['majority_accuracy']:.3f}",
            flush=True,
        )
    if not ok:
        print("STOP: Decaf did not recover FedAvg presence. FedLC is not interpreted.")
    print(f"wrote {output.resolve()}")


def main_proportions():
    import pandas as pd

    scored = proportion_table()
    if not scored:
        print("no shadow caches with a positive-sum column")
        return
    ok = fedavg_proportion_ok(scored)
    print("FedAvg proportion control", "pass" if ok else "FAIL")
    frame = pd.read_csv("results/tables/decaf.csv")
    for row in scored:
        mask = (frame["dataset"] == row["dataset"]) & (frame["method"] == row["method"])
        for key, value in row.items():
            if key in ("dataset", "method"):
                continue
            frame.loc[mask, key] = value
        print(
            f"  {row['dataset']} {row['method']} R² {row['proportion_r2']:+.3f} "
            f"[{row['proportion_r2_low']:+.3f}, {row['proportion_r2_high']:+.3f}] "
            f"pearson {row['proportion_pearson']:+.3f} "
            f"loo {row['proportion_loo_r2']:+.3f}",
            flush=True,
        )
    if not ok:
        print("STOP: Decaf proportions did not recover FedAvg. FedLC proportions are not interpreted.")
        fedavg = frame["method"] == "FedAvg"
        for column in frame.columns:
            if column.startswith("proportion") and frame.loc[~fedavg, column].notna().any():
                frame.loc[~fedavg, column] = np.nan
    frame.to_csv("results/tables/decaf.csv", index=False)
    print("updated results/tables/decaf.csv")


if __name__ == "__main__":
    import sys
    if "--proportions" in sys.argv:
        main_proportions()
    else:
        main()
