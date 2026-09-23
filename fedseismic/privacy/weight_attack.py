"""Last-layer delta attack, calibrated on shadow clients of the same method.

Features are the change in classifier row norms and the bias change, local
minus global. They are not count-sketched and they are not mixed with the rest
of the network. The ridge map is fit on simulated Dirichlet clients and applied
to the saved real uploads.
"""

from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from fedseismic.config import RunConfig, resolve_device
from fedseismic.data import build_medmnist_loaders, load_medmnist, partition_dirichlet
from fedseismic.privacy.debiased_softmax import (
    _mean, _presence_auc, _spearman, _topk, _trainer,
)
from fedseismic.privacy.estimators import clone_model_from_state, extract_last_linear
from fedseismic.privacy.log import load_global_state, load_labels, load_local_states, load_meta
from fedseismic.privacy.metrics import leakage_from_tv, to_simplex, total_variation


def layer_delta(local_state, global_state, num_classes):
    """Concatenate per-class row-norm change and bias change."""
    local_weight, local_bias = extract_last_linear(local_state)
    global_weight, global_bias = extract_last_linear(global_state)
    if local_weight is None or global_weight is None:
        return None
    delta = np.asarray(local_weight, dtype=np.float64) - np.asarray(global_weight, dtype=np.float64)
    if delta.ndim == 2:
        norms = np.linalg.norm(delta, axis=1)
    else:
        norms = np.linalg.norm(delta.reshape(delta.shape[0], -1), axis=1)
    if local_bias is None or global_bias is None:
        bias = np.zeros(num_classes, dtype=np.float64)
    else:
        bias = np.asarray(local_bias, dtype=np.float64).reshape(-1) - np.asarray(global_bias, dtype=np.float64).reshape(-1)
    if len(norms) < num_classes or len(bias) < num_classes:
        return None
    return np.concatenate([norms[:num_classes], bias[:num_classes]])


def score_run(run_dir, device="cuda", n_shadows=40):
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    device = resolve_device(device or cfg.device)
    histograms, _ = load_labels(run_dir)
    local_states = load_local_states(run_dir)
    global_state = load_global_state(run_dir)

    from fedseismic.experiment import build_federated_run

    data = build_federated_run(cfg, meta.get("seed", cfg.seed))
    images, targets, _, _, spec = load_medmnist(cfg.dataset, cfg.data_root, download=False)
    rng = np.random.RandomState(20_000 + int(meta.get("seed", cfg.seed)))
    partitions = [
        indices for indices in partition_dirichlet(targets, n_shadows, cfg.partition_alpha, rng)
        if len(indices) >= 2
    ]
    loaders = build_medmnist_loaders(
        images, targets, partitions, cfg.batch_size, spec["n_channels"], train=True,
    )

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
        feature = layer_delta(trainer.upload(), global_state, cfg.num_classes)
        if feature is None:
            continue
        features.append(feature)
        targets_pi.append(pi)

    regressor = Ridge(alpha=1.0)
    regressor.fit(np.stack(features), np.stack(targets_pi))
    null_hat = to_simplex(regressor.predict(np.zeros((1, features[0].size)))[0])

    leaks = []
    null_leaks = []
    ranks = []
    topks = []
    hats = []
    truths = []
    for client in sorted(set(histograms) & set(local_states)):
        pi = to_simplex(histograms[client])
        feature = layer_delta(local_states[client], global_state, cfg.num_classes)
        if feature is None:
            continue
        hat = to_simplex(regressor.predict(feature.reshape(1, -1))[0])
        hats.append(hat)
        truths.append(pi)
        leaks.append(leakage_from_tv(total_variation(pi, hat)))
        null_leaks.append(leakage_from_tv(total_variation(pi, null_hat)))
        ranks.append(_spearman(pi, hat))
        topks.append(_topk(pi, hat))

    attack = _mean(leaks)
    baseline = _mean(null_leaks)
    utility = meta.get("utility") or {}
    return {
        "run_dir": str(run_dir),
        "algorithm": cfg.algorithm,
        "lambda_cap": None if cfg.algorithm != "fedkper" else cfg.lambda_cap,
        "seed": meta.get("seed", cfg.seed),
        "n_shadows": len(features),
        "local_mean": utility.get("local_mean"),
        "weight_leakage": attack,
        "weight_baseline": baseline,
        "weight_advantage": attack - baseline,
        "weight_spearman": _mean(ranks),
        "weight_topk": _mean(topks),
        "weight_presence_auc": _presence_auc(hats, truths),
    }
