"""Bias and row-norm channels from the saved classifier delta.

Each file in ``head/`` is the upload minus the global that client received.
A client's score is the mean over its participations. Row norms use the size
of each row. The bias score keeps its sign: under SGD it moves up with the count.
"""

from pathlib import Path
import json

import numpy as np

from fedseismic.privacy.log import load_labels
from fedseismic.privacy.metrics import to_simplex
from fedseismic.privacy.row_delta import _summarize

RUNS = {
    ("BloodMNIST", "FedAvg"): "results/runs/channels_blood_fedavg/fedavg_noniid_a0.1_lNA_dna",
    ("BloodMNIST", "FedLC"): "results/runs/channels_blood_fedlc/fedavg_noniid_a0.1_lNA_dna",
    ("BloodMNIST", "defense"): "results/runs/channels_blood_defense/defense_noniid_a0.1_lNA_dna",
}
SEEDS = ("seed_0", "seed_1", "seed_42")


def _client_id(path):
    return int(path.stem.split("_")[-1])


def mean_channels(run_dir):
    norms = {}
    bias = {}
    folder = Path(run_dir) / "head"
    paths = sorted(folder.glob("round_*_client_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no head deltas in {folder}")
    for path in paths:
        with np.load(path) as handle:
            weight = np.asarray(handle["weight_delta"], dtype=np.float64)
            bias_delta = np.asarray(handle["bias_delta"], dtype=np.float64).reshape(-1)
        client = _client_id(path)
        norms.setdefault(client, []).append(np.linalg.norm(weight.reshape(weight.shape[0], -1), axis=1))
        bias.setdefault(client, []).append(bias_delta)
    return (
        {client: np.mean(np.stack(rows), axis=0) for client, rows in norms.items()},
        {client: np.mean(np.stack(rows), axis=0) for client, rows in bias.items()},
        {client: len(rows) for client, rows in bias.items()},
    )


SHARE_LEVELS = (0.01, 0.05, 0.10, 0.25)
ABSENT = 1e-3


def _band_auc(truths, scores, high):
    """AUC of absent against clients whose share of that class is at least ``high``."""
    from sklearn.metrics import roc_auc_score

    labels = []
    values = []
    for truth, score in zip(truths, scores):
        fraction = to_simplex(truth)
        for share, value in zip(fraction, np.asarray(score, dtype=np.float64)):
            if share <= ABSENT:
                labels.append(0)
                values.append(value)
            elif share >= high:
                labels.append(1)
                values.append(value)
    labels = np.asarray(labels)
    if len(labels) < 4 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, values))


def _client_rows(run_dir):
    histograms, _ = load_labels(run_dir)
    norms, bias, _ = mean_channels(run_dir)
    clients = sorted(client for client in bias if client in histograms)
    truths = [to_simplex(histograms[client]) for client in clients]
    return {
        "bias_mean": (truths, [bias[client] for client in clients]),
        "row_norm": (truths, [norms[client] for client in clients]),
    }


def _percentile_interval(samples):
    finite = np.asarray(samples, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_clients(truths, scores, n_draws=1000, rng=None):
    """Resample clients and recompute Pearson, presence AUC, and leave-one-out R²."""
    rng = np.random.RandomState(0) if rng is None else rng
    n_clients = len(truths)
    collected = {"pearson": [], "presence_auc": [], "loo_r2": []}
    for _ in range(n_draws):
        index = rng.randint(0, n_clients, size=n_clients)
        summary = _summarize(
            [truths[item] for item in index],
            [scores[item] for item in index],
        )
        for key in collected:
            collected[key].append(summary[key])
    return {key: _percentile_interval(values) for key, values in collected.items()}


def score_seed(run_dir):
    histograms, _ = load_labels(run_dir)
    norms, bias, counts = mean_channels(run_dir)
    clients = sorted(client for client in bias if client in histograms)
    truths = [to_simplex(histograms[client]) for client in clients]
    return {
        "bias_mean": _summarize(truths, [bias[client] for client in clients]),
        "row_norm": _summarize(truths, [norms[client] for client in clients]),
        "participations": float(np.mean([counts[client] for client in clients])),
        "local_mean": float(json.loads((Path(run_dir) / "results.json").read_text())["local_mean"]),
    }


def main():
    import pandas as pd

    rows = []
    for (dataset, method), relative in RUNS.items():
        for seed in SEEDS:
            summary = score_seed(Path(relative) / seed)
            participations = summary.pop("participations")
            local_mean = summary.pop("local_mean")
            for attack, scored in summary.items():
                row = {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "attack": attack,
                    "participations": participations,
                    "local_mean": local_mean,
                    **scored,
                }
                rows.append(row)
                print(
                    f"{dataset} {method} {seed} {attack} "
                    f"pearson {scored['pearson']:+.3f} auc {scored['presence_auc']:.3f} "
                    f"loo {scored['loo_r2']:+.3f} local {local_mean:.3f}",
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    output = Path("results/tables/head_channels.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    grouped = frame.groupby(["dataset", "method", "attack"])[
        ["pearson", "presence_auc", "loo_r2", "local_mean"]
    ].mean()
    print(grouped.to_string())
    print(f"wrote {output.resolve()}")


def write_intervals(n_draws=1000):
    """Client bootstrap and the fixed-ratio presence game on the saved Blood runs."""
    import pandas as pd

    rows = []
    for (dataset, method), relative in RUNS.items():
        for seed in SEEDS:
            channels = _client_rows(Path(relative) / seed)
            for attack, (truths, scores) in channels.items():
                point = _summarize(truths, scores)
                interval = bootstrap_clients(truths, scores, n_draws=n_draws)
                row = {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "attack": attack,
                    "n_clients": len(truths),
                }
                for key, (low, high) in interval.items():
                    row[key] = point[key]
                    row[f"{key}_low"] = low
                    row[f"{key}_high"] = high
                for level in SHARE_LEVELS:
                    row[f"share_{level:g}"] = _band_auc(truths, scores, level)
                rows.append(row)
                print(
                    f"{method} {seed} {attack} auc {point['presence_auc']:.3f} "
                    f"[{interval['presence_auc'][0]:.3f}, {interval['presence_auc'][1]:.3f}]",
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    output = Path("results/tables/head_channels_bootstrap.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"wrote {output.resolve()}")
    return frame


if __name__ == "__main__":
    main()
