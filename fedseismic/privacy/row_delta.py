"""Method-aware read of the saved upload.

The classifier-row score projects the change in each class's row onto the
public feature direction of that class. FedLC's bias update cancels, and this
term still scales with the class count. FedPer puts the global head back, so
the same score is not its upload. For that upload the score is the shift of
each class's mean feature in the body.

The baseline global is ``global_final.pt``. The global weights the client
actually received were not saved.
"""

from pathlib import Path
import json

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import pearsonr

from fedseismic.config import RunConfig, resolve_device
from fedseismic.experiment import build_federated_run
from fedseismic.privacy.distribution import histogram_r2
from fedseismic.privacy.estimators import clone_model_from_state, extract_last_linear
from fedseismic.privacy.log import load_global_state, load_labels, load_local_states, load_meta
from fedseismic.privacy.metrics import to_simplex

RUNS = {
    ("BloodMNIST", "FedAvg"): "results/runs/privacy_rounds/fedavg_noniid_a0.1_lNA_dna",
    ("BloodMNIST", "FedLC"): "results/runs/privacy_logit/fedavg_noniid_a0.1_lNA_dna",
    ("BloodMNIST", "FedPer"): "results/runs/privacy_fedper/fedper_noniid_a0.1_lNA_dna",
    ("BloodMNIST", "FedRoD"): "results/runs/fedrod_blood/fedrod_noniid_a0.1_lNA_dna",
    ("OrganCMNIST", "FedAvg"): "results/runs/privacy_rounds_organc/fedavg_noniid_a0.1_lNA_dna",
    ("OrganCMNIST", "FedLC"): "results/runs/privacy_logit_organc/fedavg_noniid_a0.1_lNA_dna",
    ("OrganCMNIST", "FedPer"): "results/runs/privacy_fedper_organc/fedper_noniid_a0.1_lNA_dna",
    ("OrganCMNIST", "FedRoD"): "results/runs/fedrod_organc/fedrod_noniid_a0.1_lNA_dna",
    ("CIFAR-10", "FedAvg"): "results/runs/privacy_cifar/fedavg_noniid_a0.1_lNA_dna",
    ("CIFAR-10", "FedLC"): "results/runs/privacy_cifar_fedlc/fedavg_noniid_a0.1_lNA_dna",
    ("CIFAR-10", "FedPer"): "results/runs/privacy_cifar/fedper_noniid_a0.1_lNA_dna",
    ("CIFAR-10", "FedRoD"): "results/runs/fedrod_cifar/fedrod_noniid_a0.1_lNA_dna",
}
SEEDS = ("seed_0", "seed_1", "seed_42")
BODY_METHODS = {"FedAvg", "FedPer", "FedLC"}


def _feature_table(model, loader, device):
    features = []
    labels = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, targets, *_ in loader:
            images = images.to(device, dtype=torch.float)
            _logits, hidden = model(images, get_features=True)
            features.append(hidden.detach().cpu())
            labels.append(targets.detach().cpu().view(-1).long())
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def _class_means(features, labels, num_classes):
    overall = features.mean(axis=0)
    means = np.zeros((num_classes, features.shape[1]), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.float64)
    for class_index in range(num_classes):
        mask = labels == class_index
        counts[class_index] = int(mask.sum())
        if mask.any():
            means[class_index] = features[mask].mean(axis=0)
    return overall, means, counts


def row_projection(local_state, global_state, overall, class_means):
    """Signed count estimate from the classifier-row change."""
    local_weight, _ = extract_last_linear(local_state)
    global_weight, _ = extract_last_linear(global_state)
    delta = np.asarray(local_weight, dtype=np.float64) - np.asarray(global_weight, dtype=np.float64)
    if delta.ndim != 2:
        raise ValueError(f"expected a linear classifier, got shape {delta.shape}")
    scores = np.zeros(delta.shape[0], dtype=np.float64)
    for class_index in range(delta.shape[0]):
        direction = class_means[class_index] - overall
        denom = float(np.dot(direction, direction))
        if denom < 1e-12:
            continue
        scores[class_index] = float(np.dot(delta[class_index], direction) / denom)
    return scores


def body_shift(local_means, global_means):
    """How far each class's public feature mean moved in the uploaded body."""
    return np.linalg.norm(local_means - global_means, axis=1)


def _population_r2(truths, predicted):
    """1 minus error relative to predicting the population mix. 0 is that guess."""
    left = np.concatenate([to_simplex(row) for row in truths])
    right = np.concatenate([to_simplex(row) for row in predicted])
    population = np.mean(np.stack([to_simplex(row) for row in truths]), axis=0)
    baseline = np.concatenate([population for _ in truths])
    denom = float(np.sum((left - baseline) ** 2))
    if denom < 1e-18:
        return 0.0
    return float(1.0 - np.sum((left - right) ** 2) / denom)


def _pearson(truths, scores):
    left = np.concatenate([to_simplex(row) for row in truths])
    right = np.concatenate([np.asarray(row, dtype=np.float64) for row in scores])
    if np.std(right) < 1e-12:
        return float("nan")
    coef = pearsonr(left, right).statistic
    return float(coef) if np.isfinite(coef) else float("nan")


def _presence_auc(truths, scores, threshold=1e-3):
    labels = []
    values = []
    for truth, score in zip(truths, scores):
        present = to_simplex(truth) > threshold
        labels.extend(present.astype(int).tolist())
        values.extend(np.asarray(score, dtype=np.float64).tolist())
    labels = np.asarray(labels)
    values = np.asarray(values)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, values))


def _loo_r2(truths, scores):
    """Leave-one-client line from the raw score to the true fraction."""
    held_true = []
    held_pred = []
    for index in range(len(truths)):
        train_x = np.concatenate([
            np.asarray(scores[other], dtype=np.float64) for other in range(len(truths)) if other != index
        ])
        train_y = np.concatenate([
            to_simplex(truths[other]) for other in range(len(truths)) if other != index
        ])
        if np.std(train_x) < 1e-12:
            prediction = np.full(len(truths[index]), train_y.mean())
        else:
            model = LinearRegression().fit(train_x.reshape(-1, 1), train_y)
            prediction = model.predict(np.asarray(scores[index], dtype=np.float64).reshape(-1, 1))
        held_true.append(to_simplex(truths[index]))
        held_pred.append(prediction)
    return float(r2_score(np.concatenate(held_true), np.concatenate(held_pred)))


def _summarize(truths, scores):
    hats = [to_simplex(np.clip(row, 0.0, None)) for row in scores]
    return {
        "r2": histogram_r2(truths, hats),
        "r2_vs_population": _population_r2(truths, hats),
        "pearson": _pearson(truths, scores),
        "presence_auc": _presence_auc(truths, scores),
        "loo_r2": _loo_r2(truths, scores),
    }


def score_seed(run_dir, device, sign, with_body):
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    data = build_federated_run(cfg, meta.get("seed", cfg.seed))
    histograms, _ = load_labels(run_dir)
    local_states = load_local_states(run_dir)
    global_state = load_global_state(run_dir)
    global_model = clone_model_from_state(data.model_factory, global_state, device=device)
    features, labels = _feature_table(global_model, data.probe_loader, device)
    overall, class_means, _ = _class_means(features, labels, cfg.num_classes)
    clients = sorted(client for client in local_states if client in histograms)
    truths = [to_simplex(histograms[client]) for client in clients]
    row_scores = [
        sign * row_projection(local_states[client], global_state, overall, class_means)
        for client in clients
    ]
    out = {"row_projection": _summarize(truths, row_scores)}
    if with_body:
        shifts = []
        for client in clients:
            model = clone_model_from_state(data.model_factory, local_states[client], device=device)
            local_features, local_labels = _feature_table(model, data.probe_loader, device)
            _, local_means, _ = _class_means(local_features, local_labels, cfg.num_classes)
            shifts.append(body_shift(local_means, class_means))
        out["body_shift"] = _summarize(truths, shifts)
    return out


def main():
    device = resolve_device("cuda")
    sign = 1.0
    probe = score_seed(Path(RUNS[("BloodMNIST", "FedAvg")]) / "seed_0", device, sign, with_body=False)
    if probe["row_projection"]["pearson"] < 0:
        sign = -1.0
    print(f"row sign {sign:+.0f} from Blood FedAvg seed 0 pearson {probe['row_projection']['pearson']:+.3f}")
    rows = []
    for (dataset, method), relative in RUNS.items():
        for seed in SEEDS:
            run_dir = Path(relative) / seed
            scored = score_seed(run_dir, device, sign, with_body=method in BODY_METHODS)
            for attack, summary in scored.items():
                row = {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "attack": attack,
                    **summary,
                }
                rows.append(row)
                print(
                    f"{dataset} {method} {seed} {attack} "
                    f"r2 {summary['r2']:+.3f} vs_pop {summary['r2_vs_population']:+.3f} "
                    f"pearson {summary['pearson']:+.3f} auc {summary['presence_auc']:.3f} "
                    f"loo {summary['loo_r2']:+.3f}",
                    flush=True,
                )
    import pandas as pd
    frame = pd.DataFrame(rows)
    output = Path("results/tables/row_delta.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"wrote {output.resolve()}")


if __name__ == "__main__":
    main()
