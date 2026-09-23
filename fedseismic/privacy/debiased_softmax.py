"""Debiased softmax attack, calibrated on shadow clients.

The feature is the held-out public softmax of a local model minus the same
softmax of the global model. A ridge map from that residual to a histogram is
fit on simulated Dirichlet clients trained for one round from that run's global
checkpoint, with the same algorithm and λ. Real clients are scored only after
that fit.
"""

from pathlib import Path

import numpy as np
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from fedseismic.config import RunConfig, resolve_device
from fedseismic.data import MEDMNIST_SPECS, build_medmnist_loaders, load_medmnist, partition_dirichlet
from fedseismic.federated.client import ClientTrainer, FedProxClientTrainer
from fedseismic.federated.fedkper import FedKPerClientTrainer
from fedseismic.federated.fedper import FedPerClientTrainer
from fedseismic.privacy.estimators import clone_model_from_state
from fedseismic.privacy.fair_leakage import _split_softmax
from fedseismic.privacy.log import load_global_state, load_labels, load_local_states, load_meta
from fedseismic.privacy.metrics import leakage_from_tv, to_simplex, total_variation


def _spearman(left, right):
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    coef = spearmanr(left, right).statistic
    return float(coef) if np.isfinite(coef) else float("nan")


def _mean(values):
    values = [value for value in values if value == value]
    if not values:
        return float("nan")
    return float(np.mean(values))


def _trainer(cfg, model, loader, device):
    common = dict(
        model=model, loader=loader, criterion=nn.CrossEntropyLoss(), device=device,
        local_epochs=cfg.local_epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
        optimizer=cfg.optimizer, momentum=cfg.momentum,
    )
    if cfg.algorithm == "fedkper":
        return FedKPerClientTrainer(**common, lambda_cap=cfg.lambda_cap, grad_clip=cfg.grad_clip)
    if cfg.algorithm == "fedprox":
        return FedProxClientTrainer(**common, mu=cfg.mu)
    if cfg.algorithm == "fedper":
        return FedPerClientTrainer(**common)
    return ClientTrainer(**common)


def _presence_auc(scores, histograms, threshold=1e-3):
    truth = (np.stack(histograms) > threshold).astype(np.int32)
    matrix = np.stack(scores)
    aucs = []
    for class_index in range(truth.shape[1]):
        labels = truth[:, class_index]
        if len(np.unique(labels)) < 2:
            continue
        aucs.append(float(roc_auc_score(labels, matrix[:, class_index])))
    return _mean(aucs)


def _topk(pi, hat, k=3):
    k = min(k, len(pi))
    true = set(np.argsort(pi)[-k:])
    pred = set(np.argsort(hat)[-k:])
    return len(true & pred) / k


def score_run(run_dir, device="cuda", n_shadows=40):
    """Calibrated residual attack for one saved seed."""
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    device = resolve_device(device or cfg.device)
    histograms, _ = load_labels(run_dir)
    local_states = load_local_states(run_dir)
    global_state = load_global_state(run_dir)
    split_at = 0 if cfg.privacy_probe_batches is None else int(cfg.privacy_probe_batches)

    from fedseismic.experiment import build_federated_run

    data = build_federated_run(cfg, meta.get("seed", cfg.seed))
    probe = data.probe_loader
    images, targets, _, _, spec = load_medmnist(cfg.dataset, cfg.data_root, download=False)
    rng = np.random.RandomState(10_000 + int(meta.get("seed", cfg.seed)))
    partitions = [
        indices for indices in partition_dirichlet(targets, n_shadows, cfg.partition_alpha, rng)
        if len(indices) >= 2
    ]
    loaders = build_medmnist_loaders(
        images, targets, partitions, cfg.batch_size, spec["n_channels"], train=True,
    )
    global_model = clone_model_from_state(data.model_factory, global_state, device=device)
    _, global_late, _, _ = _split_softmax(global_model, probe, cfg.num_classes, device, split_at)

    features = []
    targets_pi = []
    for loader in loaders:
        if len(loader.dataset) < 2:
            continue
        pi = to_simplex(np.bincount(
            np.asarray(loader.dataset.targets)[list(loader.dataset.indices)],
            minlength=cfg.num_classes,
        ).astype(np.float64))
        trainer = _trainer(cfg, data.model_factory(), loader, device)
        trainer.download(global_state)
        trainer.train(global_state=global_state)
        _, late, _, _ = _split_softmax(trainer.model, probe, cfg.num_classes, device, split_at)
        features.append(late - global_late)
        targets_pi.append(pi)

    regressor = Ridge(alpha=1.0)
    regressor.fit(np.stack(features), np.stack(targets_pi))

    hats = []
    truths = []
    leaks = []
    ranks = []
    topks = []
    raw_leaks = []
    for client in sorted(set(histograms) & set(local_states)):
        pi = to_simplex(histograms[client])
        local_model = clone_model_from_state(data.model_factory, local_states[client], device=device)
        _, late, _, _ = _split_softmax(local_model, probe, cfg.num_classes, device, split_at)
        hat = to_simplex(regressor.predict((late - global_late).reshape(1, -1))[0])
        hats.append(hat)
        truths.append(pi)
        leaks.append(leakage_from_tv(total_variation(pi, hat)))
        ranks.append(_spearman(pi, hat))
        topks.append(_topk(pi, hat))
        raw_leaks.append(leakage_from_tv(total_variation(pi, late)))

    utility = meta.get("utility") or {}
    return {
        "run_dir": str(run_dir),
        "algorithm": cfg.algorithm,
        "lambda_cap": None if cfg.algorithm != "fedkper" else cfg.lambda_cap,
        "seed": meta.get("seed", cfg.seed),
        "n_shadows": len(features),
        "local_mean": utility.get("local_mean"),
        "raw_leakage": _mean(raw_leaks),
        "debiased_leakage": _mean(leaks),
        "debiased_spearman": _mean(ranks),
        "debiased_topk": _mean(topks),
        "debiased_presence_auc": _presence_auc(hats, truths),
    }
