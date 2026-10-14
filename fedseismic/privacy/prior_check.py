"""Checks that do not need a new training run.

Logit adjustment asks whether the local-accuracy gap is just the client prior.
Rank correlation asks whether distillation removed that prior or only shrank it.
"""

from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from fedseismic.config import RunConfig, resolve_device
from fedseismic.eval.metrics import _logits, score_loader
from fedseismic.privacy.estimators import clone_model_from_state, softmax_prior
from fedseismic.privacy.log import load_global_state, load_labels, load_local_states, load_meta
from fedseismic.privacy.metrics import to_simplex, total_variation


def _client_accuracy(model, loader, num_classes, device, log_prior=None):
    if log_prior is None:
        return score_loader(model, loader, num_classes, device, task="classification")[0]
    model.eval()
    model.to(device)
    offset = torch.as_tensor(log_prior, device=device, dtype=torch.float)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets, *_ in loader:
            images = images.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)
            logits = _logits(model(images))
            if logits.ndim == 4:
                logits = logits + offset.view(1, -1, 1, 1)
            else:
                logits = logits + offset.view(1, -1)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += int(targets.numel())
    if total == 0:
        return float("nan")
    return correct / total


def _mean_spearman(pairs):
    values = []
    for left, right in pairs:
        if np.std(left) < 1e-12 or np.std(right) < 1e-12:
            continue
        coef = spearmanr(left, right).statistic
        if np.isfinite(coef):
            values.append(float(coef))
    if not values:
        return float("nan")
    return float(np.mean(values))


def _presence_auc(scores_by_client, histograms, threshold=1e-3):
    clients = sorted(scores_by_client)
    if len(clients) < 2:
        return float("nan")
    matrix = np.stack([scores_by_client[client] for client in clients])
    truth = np.stack([(histograms[client] > threshold).astype(np.int32) for client in clients])
    aucs = []
    for class_index in range(truth.shape[1]):
        labels = truth[:, class_index]
        if len(np.unique(labels)) < 2:
            continue
        aucs.append(float(roc_auc_score(labels, matrix[:, class_index])))
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def check_run(run_dir, device="cuda"):
    """Score one saved seed. Uses true π only for the logit-adjustment test."""
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    device = resolve_device(device or cfg.device)
    histograms, _ = load_labels(run_dir)
    local_states = load_local_states(run_dir)
    global_state = load_global_state(run_dir)

    from fedseismic.experiment import build_federated_run

    data = build_federated_run(cfg, meta.get("seed", cfg.seed))
    probe = data.probe_loader
    population = np.bincount(data.train_labels, minlength=cfg.num_classes).astype(np.float64)
    population = to_simplex(population)
    clients = sorted(set(histograms) & set(local_states))

    global_model = clone_model_from_state(data.model_factory, global_state, device=device)
    global_softmax = softmax_prior(
        global_model, probe, cfg.num_classes, device, cfg.privacy_probe_batches,
    )
    local_accs = []
    global_accs = []
    adjusted_accs = []
    prior_tvs = []
    raw_pairs = []
    residual_pairs = []
    local_softmax = {}
    residual = {}
    for client in clients:
        loader = data.client_tests[client]
        if loader is None or not len(loader.dataset):
            continue
        pi = to_simplex(histograms[client])
        local_model = clone_model_from_state(data.model_factory, local_states[client], device=device)
        local_accs.append(_client_accuracy(local_model, loader, cfg.num_classes, device))
        global_accs.append(_client_accuracy(global_model, loader, cfg.num_classes, device))
        adjusted_accs.append(_client_accuracy(
            global_model, loader, cfg.num_classes, device, log_prior=np.log(np.clip(pi, 1e-8, None)),
        ))
        hat = softmax_prior(
            local_model, probe, cfg.num_classes, device, cfg.privacy_probe_batches,
        )
        prior_tvs.append(total_variation(pi, population))
        raw_pairs.append((pi, hat))
        residual_pairs.append((pi - population, hat - global_softmax))
        local_softmax[client] = hat
        residual[client] = hat - global_softmax

    local_mean = float(np.mean(local_accs))
    global_mean = float(np.mean(global_accs))
    adjusted_mean = float(np.mean(adjusted_accs))
    gap = local_mean - global_mean
    closed = float("nan") if abs(gap) < 1e-8 else (adjusted_mean - global_mean) / gap
    worst = np.sort(np.asarray(local_accs))
    tail = max(1, int(np.ceil(0.1 * len(worst))))
    return {
        "run_dir": str(run_dir),
        "algorithm": cfg.algorithm,
        "lambda_cap": cfg.lambda_cap,
        "fedkper_diversity": cfg.fedkper_diversity,
        "seed": meta.get("seed", cfg.seed),
        "local_mean": local_mean,
        "worst10": float(np.mean(worst[:tail])),
        "global_on_local": global_mean,
        "global_plus_log_pi": adjusted_mean,
        "gap_closed_by_prior": closed,
        "prior_only_leakage": float(np.mean([1.0 - tv for tv in prior_tvs])),
        "softmax_spearman": _mean_spearman(raw_pairs),
        "residual_spearman": _mean_spearman(residual_pairs),
        "presence_auc": _presence_auc(local_softmax, histograms),
        "residual_presence_auc": _presence_auc(residual, histograms),
    }
