"""Attacks that can be scored from the files already on disk.

Bias change uses ``round_scores.jsonl``. Each line is upload minus the global
that client received, and a client's score is the mean over its participations.
Row norms are ``||W_upload - W_final||`` per class. The reference there is
still ``global_final.pt``.

The body probe freezes the uploaded body, fits a balanced logistic regression
on one half of the public probe, and reads per-class recall and margin on the
other half. Class difficulty is removed by subtracting the mean over clients.
FedAvg is the check: a FedPer number is meaningful only if FedAvg is readable.
"""

from pathlib import Path
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from fedseismic.config import RunConfig, resolve_device
from fedseismic.experiment import build_federated_run
from fedseismic.privacy.estimators import clone_model_from_state, extract_last_linear
from fedseismic.privacy.log import load_global_state, load_labels, load_local_states, load_meta
from fedseismic.privacy.metrics import to_simplex
from fedseismic.privacy.row_delta import RUNS, SEEDS, _feature_table, _summarize

PROBE_METHODS = {"FedAvg", "FedPer"}


def mean_bias(run_dir):
    """Mean received-referenced bias change, one vector per client."""
    buckets = {}
    path = Path(run_dir) / "round_scores.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("bias_delta") is None:
                continue
            buckets.setdefault(int(record["client"]), []).append(
                np.asarray(record["bias_delta"], dtype=np.float64)
            )
    means = {client: np.mean(np.stack(rows), axis=0) for client, rows in buckets.items()}
    counts = {client: len(rows) for client, rows in buckets.items()}
    return means, counts


def row_norms(local_state, global_state):
    """Size of each classifier-row change. Larger means more of that class."""
    local_weight, _ = extract_last_linear(local_state)
    global_weight, _ = extract_last_linear(global_state)
    delta = np.asarray(local_weight, dtype=np.float64) - np.asarray(global_weight, dtype=np.float64)
    return np.linalg.norm(delta.reshape(delta.shape[0], -1), axis=1)


def _class_scores(decision, predicted, labels, num_classes):
    recalls = np.full(num_classes, np.nan, dtype=np.float64)
    margins = np.full(num_classes, np.nan, dtype=np.float64)
    for class_index in range(num_classes):
        mask = labels == class_index
        if not np.any(mask):
            continue
        recalls[class_index] = float(np.mean(predicted[mask] == class_index))
        chosen = decision[mask, class_index]
        others = np.max(np.delete(decision[mask], class_index, axis=1), axis=1)
        margins[class_index] = float(np.mean(chosen - others))
    return recalls, margins


def _center_clients(scores):
    stacked = np.vstack(scores)
    if not np.isfinite(stacked).all():
        raise ValueError("probe score has a missing class on the scoring half")
    return stacked - stacked.mean(axis=0)


def body_probe(run_dir, device):
    """Centered per-class recall and margin of a probe on the uploaded body."""
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    data = build_federated_run(cfg, meta.get("seed", cfg.seed))
    histograms, _ = load_labels(run_dir)
    local_states = load_local_states(run_dir)
    clients = sorted(client for client in local_states if client in histograms)
    truths = [to_simplex(histograms[client]) for client in clients]
    recalls = []
    margins = []
    split = None
    for client in clients:
        model = clone_model_from_state(data.model_factory, local_states[client], device=device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        features, labels = _feature_table(model, data.probe_loader, device)
        if split is None:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
            fit_index, score_index = next(splitter.split(np.zeros(len(labels)), labels))
            split = (fit_index, score_index)
        fit_index, score_index = split
        scaler = StandardScaler()
        fit_x = scaler.fit_transform(features[fit_index])
        score_x = scaler.transform(features[score_index])
        classifier = LogisticRegression(
            class_weight="balanced", max_iter=500, solver="lbfgs",
        )
        classifier.fit(fit_x, labels[fit_index])
        decision = np.asarray(classifier.decision_function(score_x), dtype=np.float64)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        predicted = np.asarray(classifier.predict(score_x))
        recall, margin = _class_scores(decision, predicted, labels[score_index], cfg.num_classes)
        recalls.append(recall)
        margins.append(margin)
    return {
        "body_recall": _summarize(truths, _center_clients(recalls)),
        "body_margin": _summarize(truths, _center_clients(margins)),
    }


def weight_checks(run_dir):
    run_dir = Path(run_dir)
    histograms, _ = load_labels(run_dir)
    bias, counts = mean_bias(run_dir)
    local_states = load_local_states(run_dir)
    global_state = load_global_state(run_dir)
    clients = sorted(client for client in bias if client in histograms and client in local_states)
    truths = [to_simplex(histograms[client]) for client in clients]
    bias_scores = [bias[client] for client in clients]
    norms = [row_norms(local_states[client], global_state) for client in clients]
    return {
        "bias_mean": _summarize(truths, bias_scores),
        "row_norm": _summarize(truths, norms),
        "participations": float(np.mean([counts[client] for client in clients])),
    }


def main():
    device = resolve_device("cuda")
    rows = []
    for (dataset, method), relative in RUNS.items():
        for seed in SEEDS:
            run_dir = Path(relative) / seed
            scored = weight_checks(run_dir)
            participations = scored.pop("participations")
            if method in PROBE_METHODS:
                print(f"{dataset} {method} {seed} probe", flush=True)
                scored.update(body_probe(run_dir, device))
            for attack, summary in scored.items():
                row = {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "attack": attack,
                    "participations": participations,
                    **summary,
                }
                rows.append(row)
                print(
                    f"{dataset} {method} {seed} {attack} "
                    f"pearson {summary['pearson']:+.3f} auc {summary['presence_auc']:.3f} "
                    f"loo {summary['loo_r2']:+.3f} r2 {summary['r2']:+.3f} "
                    f"parts {participations:.2f}",
                    flush=True,
                )
    import pandas as pd
    frame = pd.DataFrame(rows)
    output = Path("results/tables/disk_checks.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"wrote {output.resolve()}")


if __name__ == "__main__":
    main()
