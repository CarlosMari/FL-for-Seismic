"""Mechanism checks on the saved prior-smoothing runs.

The slope test pools the bias change, upload minus the model the client
received, against the class share. The three-dataset table resamples clients
inside each seed. Neither fit uses a victim's labels to train an attacker.
"""

from pathlib import Path
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from fedseismic.privacy.head_channels import mean_channels
from fedseismic.privacy.log import load_labels
from fedseismic.privacy.meta_attacker import SHARE_BINS, in_share_bin
from fedseismic.privacy.metrics import to_simplex
from fedseismic.privacy.row_delta import _pearson, _presence_auc, _summarize

ROOTS = {
    "bloodmnist": Path("results/runs/prior_smooth_blood"),
    "organcmnist": Path("results/runs/prior_smooth_organc"),
    "cifar10": Path("results/runs/prior_smooth_cifar"),
}
DISPLAY = {
    "bloodmnist": "Blood",
    "organcmnist": "OrganC",
    "cifar10": "CIFAR-10",
}
ABSENT = 1e-3
# Point estimates already printed in findings.md. A larger gap is a changed claim.
PUBLISHED_AUC = {
    ("Blood", "FedLC", "row"): 0.98,
    ("Blood", "FedLC", "bias"): 0.50,
    ("Blood", "FedAvg", "row"): 0.70,
    ("Blood", "FedAvg", "bias"): 0.82,
    ("OrganC", "FedLC", "row"): 1.00,
    ("OrganC", "FedLC", "bias"): 0.47,
    ("OrganC", "FedAvg", "row"): 0.73,
    ("OrganC", "FedAvg", "bias"): 0.82,
    ("CIFAR-10", "FedLC", "row"): 0.99,
    ("CIFAR-10", "FedLC", "bias"): 0.53,
    ("CIFAR-10", "FedAvg", "row"): 0.72,
    ("CIFAR-10", "FedAvg", "bias"): 0.87,
}
PUBLISHED_LOCAL = {
    ("Blood", "FedLC"): 0.932,
    ("Blood", "FedAvg"): 0.905,
    ("OrganC", "FedLC"): 0.935,
    ("OrganC", "FedAvg"): 0.917,
    ("CIFAR-10", "FedLC"): 0.886,
    ("CIFAR-10", "FedAvg"): 0.852,
}


def _percentile(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def load_client_rows(run_dir):
    """One row per client: true share, mean bias change, mean row size."""
    histograms, _ = load_labels(run_dir)
    norms, bias, _ = mean_channels(run_dir)
    rows = []
    for client in sorted(bias):
        if client not in histograms:
            continue
        rows.append({
            "client": int(client),
            "pi": to_simplex(histograms[client]),
            "bias": np.asarray(bias[client], dtype=np.float64),
            "row": np.asarray(norms[client], dtype=np.float64),
        })
    if not rows:
        raise FileNotFoundError(f"no client channels in {run_dir}")
    return rows


def pooled_slope(rows, key="bias"):
    """Ordinary-least-squares slope of one channel on π, pooled over clients and classes."""
    left = np.concatenate([row["pi"] for row in rows])
    right = np.concatenate([np.asarray(row[key], dtype=np.float64) for row in rows])
    centered = left - left.mean()
    denom = float(np.dot(centered, centered))
    if denom < 1e-18:
        return float("nan")
    return float(np.dot(centered, right - right.mean()) / denom)


def pooled_intercept(rows, key="bias"):
    left = np.concatenate([row["pi"] for row in rows])
    right = np.concatenate([np.asarray(row[key], dtype=np.float64) for row in rows])
    slope = pooled_slope(rows, key)
    if not np.isfinite(slope):
        return float("nan")
    return float(right.mean() - slope * left.mean())


def prior_presence_auc(rows):
    """Presence AUC of an attacker that scores every client with the population mix."""
    truths = [row["pi"] for row in rows]
    population = np.mean(np.stack(truths), axis=0)
    return _presence_auc(truths, [population for _ in truths])


def band_auc(rows, key, lo, hi):
    """AUC of absent against shares inside one bin, using a saved channel as the score."""
    labels = []
    values = []
    for row in rows:
        for share, value in zip(row["pi"], np.asarray(row[key], dtype=np.float64)):
            if share <= ABSENT:
                labels.append(0)
                values.append(float(value))
            elif in_share_bin(share, lo, hi):
                labels.append(1)
                values.append(float(value))
    labels = np.asarray(labels)
    if len(labels) < 4 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, values))


def band_floor(rows, lo, hi):
    """Same bin, scored by the population share of the class."""
    truths = np.stack([row["pi"] for row in rows])
    population = truths.mean(axis=0)
    labels = []
    values = []
    for pi in truths:
        for class_index, share in enumerate(pi):
            if share <= ABSENT:
                labels.append(0)
                values.append(float(population[class_index]))
            elif in_share_bin(share, lo, hi):
                labels.append(1)
                values.append(float(population[class_index]))
    labels = np.asarray(labels)
    if len(labels) < 4 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, values))


def band_support(rows, lo, hi):
    absent = 0
    inside = 0
    for row in rows:
        for share in row["pi"]:
            if share <= ABSENT:
                absent += 1
            elif in_share_bin(share, lo, hi):
                inside += 1
    return absent, inside


def _resample(rows, rng):
    index = rng.randint(0, len(rows), size=len(rows))
    return [rows[item] for item in index]


def bootstrap_pooled(groups, statistic, n_draws=1000, rng=None):
    """Resample clients inside each seed, pool the draws, and refit one number."""
    rng = np.random.RandomState(0) if rng is None else rng
    point = float(statistic([row for group in groups for row in group]))
    draws = []
    for _ in range(n_draws):
        pooled = []
        for group in groups:
            pooled.extend(_resample(group, rng))
        draws.append(float(statistic(pooled)))
    low, high = _percentile(draws)
    return point, low, high, draws


def _mean_finite(values):
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def bootstrap_seed_mean(groups, statistic, n_draws=1000, rng=None):
    """Mean of per-seed statistics. Each draw resamples clients inside every seed."""
    rng = np.random.RandomState(0) if rng is None else rng
    point = _mean_finite(statistic(group) for group in groups)
    draws = []
    for _ in range(n_draws):
        draws.append(_mean_finite(statistic(_resample(group, rng)) for group in groups))
    low, high = _percentile(draws)
    return point, low, high, draws


def linear_fit(lambdas, slopes):
    """Slope, intercept, and R² of ``slopes`` against ``lambdas``."""
    lambdas = np.asarray(lambdas, dtype=np.float64)
    slopes = np.asarray(slopes, dtype=np.float64)
    if len(lambdas) < 2 or np.std(lambdas) < 1e-12:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(lambdas, slopes, 1)
    predicted = slope * lambdas + intercept
    residual = float(np.sum((slopes - predicted) ** 2))
    total = float(np.sum((slopes - slopes.mean()) ** 2))
    r2 = float("nan") if total < 1e-18 else float(1.0 - residual / total)
    return float(slope), float(intercept), r2


def iter_runs():
    found = []
    for dataset, root in ROOTS.items():
        if not root.is_dir():
            continue
        for job in sorted(path for path in root.iterdir() if path.is_dir()):
            for run in sorted(job.glob("seed_*")):
                meta_path = run / "run_meta.json"
                if not meta_path.is_file() or not (run / "results.json").is_file():
                    continue
                meta = json.loads(meta_path.read_text())
                config = meta["config"]
                found.append({
                    "dataset": str(config.get("dataset", dataset)),
                    "prior_smooth": float(config["prior_smooth"]),
                    "seed": run.name,
                    "num_classes": int(config["num_classes"]),
                    "path": run,
                    "local_mean": float(json.loads((run / "results.json").read_text())["local_mean"]),
                })
    if not found:
        raise FileNotFoundError("no prior-smoothing runs")
    return found


def _group_rows(runs):
    grouped = {}
    for run in runs:
        rows = load_client_rows(run["path"])
        grouped.setdefault((run["dataset"], run["prior_smooth"]), {}).setdefault(run["seed"], rows)
        run["n_clients"] = len(rows)
    return grouped


def slope_table(grouped, n_draws=1000):
    rows = []
    for (dataset, smooth), seeds in sorted(grouped.items()):
        groups = [seeds[name] for name in sorted(seeds)]
        slope, low, high, _ = bootstrap_pooled(groups, pooled_slope, n_draws=n_draws)
        intercept = pooled_intercept([row for group in groups for row in group])
        classes = int(groups[0][0]["pi"].shape[0])
        centered = intercept + slope / classes
        pearson = _pearson(
            [row["pi"] for group in groups for row in group],
            [row["bias"] for group in groups for row in group],
        )
        rows.append({
            "dataset": DISPLAY.get(dataset, dataset),
            "dataset_key": dataset,
            "lambda": smooth,
            "n_seeds": len(groups),
            "n_clients": int(sum(len(group) for group in groups)),
            "num_classes": classes,
            "slope": slope,
            "slope_low": low,
            "slope_high": high,
            "intercept": intercept,
            "centered_intercept": centered,
            "pearson": pearson,
        })
    return rows


def slope_line(grouped, n_draws=1000):
    """Line of Blood slopes against λ, with a bootstrap interval on the line's slope and R²."""
    blood = {
        smooth: seeds for (dataset, smooth), seeds in grouped.items() if dataset == "bloodmnist"
    }
    lambdas = sorted(blood)
    table = {
        item["lambda"]: item for item in slope_table(
            {("bloodmnist", smooth): seeds for smooth, seeds in blood.items()}, n_draws=n_draws,
        )
    }
    point_slopes = [table[value]["slope"] for value in lambdas]
    line_slope, line_intercept, line_r2 = linear_fit(lambdas, point_slopes)
    rng = np.random.RandomState(1)
    fit_slopes = []
    fit_r2 = []
    for _ in range(n_draws):
        drawn = []
        for value in lambdas:
            groups = [blood[value][name] for name in sorted(blood[value])]
            pooled = []
            for group in groups:
                pooled.extend(_resample(group, rng))
            drawn.append(pooled_slope(pooled))
        slope, _, r2 = linear_fit(lambdas, drawn)
        fit_slopes.append(slope)
        fit_r2.append(r2)
    slope_low, slope_high = _percentile(fit_slopes)
    r2_low, r2_high = _percentile(fit_r2)
    return {
        "line_slope": line_slope,
        "line_slope_low": slope_low,
        "line_slope_high": slope_high,
        "line_intercept": line_intercept,
        "line_r2": line_r2,
        "line_r2_low": r2_low,
        "line_r2_high": r2_high,
    }


def _channel_statistic(key):
    def statistic(rows):
        return _presence_auc([row["pi"] for row in rows], [row[key] for row in rows])
    return statistic


def three_dataset_table(grouped, runs, n_draws=1000):
    saved_local = {}
    for run in runs:
        if run["prior_smooth"] not in (0.0, 1.0):
            continue
        method = "FedLC" if run["prior_smooth"] == 0.0 else "FedAvg"
        saved_local.setdefault((DISPLAY.get(run["dataset"], run["dataset"]), method), []).append(run["local_mean"])
    rows = []
    for (dataset, smooth), seeds in sorted(grouped.items()):
        if smooth not in (0.0, 1.0):
            continue
        method = "FedLC" if smooth == 0.0 else "FedAvg"
        name = DISPLAY.get(dataset, dataset)
        groups = [seeds[seed] for seed in sorted(seeds)]
        for key, attack in (("bias", "bias"), ("row", "row")):
            estimate, low, high, _ = bootstrap_seed_mean(groups, _channel_statistic(key), n_draws=n_draws)
            floor, floor_low, floor_high, _ = bootstrap_seed_mean(
                groups, prior_presence_auc, n_draws=n_draws, rng=np.random.RandomState(0),
            )
            # Paired leakage: same resample for the attack and the floor.
            rng = np.random.RandomState(0)
            leak_draws = []
            for _ in range(n_draws):
                attack_values = []
                floor_values = []
                for group in groups:
                    sample = _resample(group, rng)
                    attack_values.append(_channel_statistic(key)(sample))
                    floor_values.append(prior_presence_auc(sample))
                leak_draws.append(float(np.mean(attack_values) - np.mean(floor_values)))
            leak_low, leak_high = _percentile(leak_draws)
            rows.append({
                "dataset": name,
                "method": method,
                "attack": attack,
                "estimate": estimate,
                "low": low,
                "high": high,
                "floor": floor,
                "floor_low": floor_low,
                "floor_high": floor_high,
                "leakage": estimate - floor,
                "leakage_low": leak_low,
                "leakage_high": leak_high,
                "n_seeds": len(groups),
            })
        local_values = saved_local[(name, method)]
        rows.append({
            "dataset": name,
            "method": method,
            "attack": "local",
            "estimate": float(np.mean(local_values)),
            "low": float("nan"),
            "high": float("nan"),
            "floor": float("nan"),
            "floor_low": float("nan"),
            "floor_high": float("nan"),
            "leakage": float("nan"),
            "leakage_low": float("nan"),
            "leakage_high": float("nan"),
            "n_seeds": len(local_values),
        })
    return rows


def hand_share_table(grouped, n_draws=1000):
    """Bin AUC of the bias change and the row size. Accuracy needs the learned attacker."""
    rows = []
    for (dataset, smooth), seeds in sorted(grouped.items()):
        if smooth not in (0.0, 1.0):
            continue
        method = "FedLC" if smooth == 0.0 else "FedAvg"
        name = DISPLAY.get(dataset, dataset)
        groups = [seeds[seed] for seed in sorted(seeds)]
        for key, attack in (("bias", "hand_bias"), ("row", "hand_row")):
            for label, lo, hi in SHARE_BINS:
                absent, inside = band_support([row for group in groups for row in group], lo, hi)

                def statistic(sample, key=key, lo=lo, hi=hi):
                    return band_auc(sample, key, lo, hi)

                def floor_statistic(sample, lo=lo, hi=hi):
                    return band_floor(sample, lo, hi)

                estimate, low, high, _ = bootstrap_seed_mean(groups, statistic, n_draws=n_draws)
                floor, _, _, _ = bootstrap_seed_mean(
                    groups, floor_statistic, n_draws=n_draws, rng=np.random.RandomState(0),
                )
                rows.append({
                    "dataset": name,
                    "method": method,
                    "attacker": attack,
                    "bin": label,
                    "auc": estimate,
                    "auc_low": low,
                    "auc_high": high,
                    "accuracy": float("nan"),
                    "floor": floor,
                    "leakage": estimate - floor,
                    "n_absent": absent,
                    "n_bin": inside,
                })
    return rows


def attach_local_intervals(table, per_client, n_draws=1000):
    """Fill local-accuracy intervals from per-client scores. ``per_client`` maps (dataset, method, seed) to a dict."""
    rng = np.random.RandomState(0)
    grouped = {}
    saved = {}
    for (dataset, method, seed), scores in per_client.items():
        grouped.setdefault((dataset, method), []).append(np.asarray(list(scores.values()), dtype=np.float64))
        saved.setdefault((dataset, method), {})[seed] = scores
    for row in table:
        if row["attack"] != "local":
            continue
        groups = grouped.get((row["dataset"], row["method"]))
        if not groups:
            continue
        point = float(np.mean([float(group.mean()) for group in groups]))
        draws = []
        for _ in range(n_draws):
            draws.append(float(np.mean([
                float(group[rng.randint(0, len(group), size=len(group))].mean()) for group in groups
            ])))
        low, high = _percentile(draws)
        row["recomputed"] = point
        row["low"] = low
        row["high"] = high
    return table


def client_local_scores(run, device):
    """Per-client holdout accuracy of the saved upload, with log π only for FedLC."""
    from fedseismic.config import RunConfig
    from fedseismic.experiment import build_federated_run
    from fedseismic.federated.server import _accuracy
    from fedseismic.privacy.estimators import clone_model_from_state
    from fedseismic.privacy.log import load_local_states, load_meta

    run_dir = Path(run["path"])
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    data = build_federated_run(cfg, meta.get("seed", cfg.seed))
    states = load_local_states(run_dir)
    histograms, _ = load_labels(run_dir)
    scores = {}
    for client, state in states.items():
        if client >= len(data.client_tests):
            continue
        loader = data.client_tests[client]
        if loader is None or not len(loader.dataset):
            continue
        log_prior = None
        if cfg.logit_adjust and client in histograms:
            log_prior = np.log(np.clip(np.asarray(histograms[client], dtype=np.float64), 1e-8, None))
        model = clone_model_from_state(data.model_factory, state, device=device)
        scores[int(client)] = float(_accuracy(model, loader, device, log_prior))
    return scores


def plot_slopes(table, output):
    import matplotlib.pyplot as plt

    frame = [row for row in table]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = {"Blood": "black", "OrganC": "C0", "CIFAR-10": "C1"}
    markers = {"Blood": "o", "OrganC": "s", "CIFAR-10": "D"}
    for dataset in ("Blood", "OrganC", "CIFAR-10"):
        part = [row for row in frame if row["dataset"] == dataset]
        if not part:
            continue
        part = sorted(part, key=lambda row: row["lambda"])
        xs = [row["lambda"] for row in part]
        ys = [row["slope"] for row in part]
        yerr = np.vstack([
            [row["slope"] - row["slope_low"] for row in part],
            [row["slope_high"] - row["slope"] for row in part],
        ])
        ax.errorbar(
            xs, ys, yerr=yerr, fmt=markers[dataset], color=colors[dataset],
            capsize=3, label=dataset,
        )
    blood = sorted([row for row in frame if row["dataset"] == "Blood"], key=lambda row: row["lambda"])
    if len(blood) >= 2:
        slope, intercept, _ = linear_fit(
            [row["lambda"] for row in blood], [row["slope"] for row in blood],
        )
        grid = np.linspace(0.0, 1.0, 50)
        ax.plot(grid, slope * grid + intercept, color="black", linewidth=1.0, linestyle="--")
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.set_xlabel("Prior smoothing λ")
    ax.set_ylabel("Slope of bias change on class share")
    ax.set_title("Bias change versus class share")
    ax.legend()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def fedavg_slope_ok(table):
    """Positive control: ordinary training (λ = 1) has a positive slope whose interval excludes 0."""
    checks = []
    for row in table:
        if abs(row["lambda"] - 1.0) > 1e-9:
            continue
        checks.append(row["slope"] > 0.0 and row["slope_low"] > 0.0)
    return bool(checks) and all(checks)


def claim_flags(table):
    flags = []
    for row in table:
        if row["attack"] == "local":
            published = PUBLISHED_LOCAL.get((row["dataset"], row["method"]))
            tolerance = 0.002
        else:
            published = PUBLISHED_AUC.get((row["dataset"], row["method"], row["attack"]))
            tolerance = 0.015
        if published is None:
            continue
        gap = abs(row["estimate"] - published)
        if gap > tolerance:
            flags.append(
                f"{row['dataset']} {row['method']} {row['attack']}: "
                f"computed {row['estimate']:.4f}, published {published:.3f}"
            )
    return flags


def write_table(rows, path):
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return path


def fill_local(n_draws=1000, device="cuda"):
    """Recompute per-client holdout accuracy and write intervals into the three-dataset table."""
    import pandas as pd

    runs = [run for run in iter_runs() if abs(run["prior_smooth"] - 0.0) < 1e-9 or abs(run["prior_smooth"] - 1.0) < 1e-9]
    per_client = {}
    for run in runs:
        name = DISPLAY.get(run["dataset"], run["dataset"])
        method = "FedLC" if abs(run["prior_smooth"]) < 1e-9 else "FedAvg"
        print(f"local {name} {method} {run['seed']}", flush=True)
        scores = client_local_scores(run, device)
        per_client[(name, method, run["seed"])] = scores
        recomputed = float(np.mean(list(scores.values()))) if scores else float("nan")
        print(f"  saved {run['local_mean']:.4f} recomputed {recomputed:.4f} n {len(scores)}", flush=True)
        relative = Path(run["path"]).resolve().relative_to(Path("results/runs").resolve())
        path = Path("results/attacks/local_accuracy") / relative / "accuracy.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        clients = np.asarray(sorted(scores), dtype=np.int32)
        np.savez_compressed(path, client=clients, accuracy=np.asarray([scores[int(c)] for c in clients]))
    frame = pd.read_csv("results/tables/three_dataset.csv")
    table = frame.to_dict(orient="records")
    attach_local_intervals(table, per_client, n_draws=n_draws)
    write_table(table, "results/tables/three_dataset.csv")
    for row in table:
        if row["attack"] != "local":
            continue
        gap = abs(float(row.get("recomputed", float("nan"))) - float(row["estimate"]))
        print(
            f"{row['dataset']} {row['method']} local {row['estimate']:.4f} "
            f"[{row['low']:.4f}, {row['high']:.4f}] recomputed gap {gap:.4f}",
            flush=True,
        )
        if gap > 0.005:
            print("FLAG recomputed local accuracy does not match the saved mean")


def main():
    runs = iter_runs()
    grouped = _group_rows(runs)
    slopes = slope_table(grouped)
    line = slope_line(grouped)
    write_table(slopes, "results/tables/lambda_slope.csv")
    plot_slopes(slopes, "results/figures/lambda_slope.png")
    control = fedavg_slope_ok(slopes)
    print("slope control", "pass" if control else "FAIL")
    for row in slopes:
        print(
            f"  {row['dataset']} λ={row['lambda']:g} slope {row['slope']:+.4f} "
            f"[{row['slope_low']:+.4f}, {row['slope_high']:+.4f}] "
            f"centered intercept {row['centered_intercept']:+.4f}",
            flush=True,
        )
    print(
        "Blood line: d(slope)/dλ "
        f"{line['line_slope']:+.4f} [{line['line_slope_low']:+.4f}, {line['line_slope_high']:+.4f}] "
        f"R² {line['line_r2']:.3f} [{line['line_r2_low']:.3f}, {line['line_r2_high']:.3f}]",
        flush=True,
    )
    (Path("results/tables/lambda_slope_line.json")).write_text(json.dumps(line, indent=2), encoding="utf-8")
    if not control:
        print("STOP: FedAvg slope control failed. Other λ values are not interpreted.")
        return
    channels = three_dataset_table(grouped, runs)
    flags = claim_flags(channels)
    write_table(channels, "results/tables/three_dataset.csv")
    shares = hand_share_table(grouped)
    write_table(shares, "results/tables/share_sweep_hand.csv")
    for row in channels:
        if row["attack"] == "local":
            print(
                f"  {row['dataset']} {row['method']} local {row['estimate']:.4f}",
                flush=True,
            )
        else:
            print(
                f"  {row['dataset']} {row['method']} {row['attack']} "
                f"{row['estimate']:.3f} [{row['low']:.3f}, {row['high']:.3f}] "
                f"floor {row['floor']:.3f} leakage {row['leakage']:+.3f}",
                flush=True,
            )
    if flags:
        print("FLAG published point estimates differ:")
        for flag in flags:
            print(" ", flag)
    else:
        print("Published point estimates match.")
    print("wrote results/tables/lambda_slope.csv results/tables/three_dataset.csv results/figures/lambda_slope.png")


if __name__ == "__main__":
    import sys
    if "--local" in sys.argv:
        fill_local()
    else:
        main()
