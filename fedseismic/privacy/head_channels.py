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


if __name__ == "__main__":
    main()
