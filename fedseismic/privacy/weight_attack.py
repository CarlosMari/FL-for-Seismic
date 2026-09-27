"""Shadow ridge attacks, calibrated on clients of the same method.

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
from fedseismic.privacy.estimators import extract_last_linear, flatten_state_delta
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


def fit_delta_map(features, targets, rank=32, alpha=1.0):
    """Ridge from a full weight delta to π, in the shadow clients' PCA basis.

    The basis and the column scales are fit on the shadows only. ``rank`` is
    capped by the number of shadows.
    """
    matrix = np.stack(features).astype(np.float64)
    target = np.stack(targets).astype(np.float64)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-8] = 1.0
    centered = (matrix - mean) / std
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    used = min(int(rank), max(1, len(matrix) - 1), vt.shape[0])
    basis = vt[:used]
    coded = centered @ basis.T
    regressor = Ridge(alpha=alpha)
    regressor.fit(coded, target)
    return {"mean": mean, "std": std, "basis": basis, "regressor": regressor}


def apply_delta_map(feature, mapping):
    coded = ((np.asarray(feature, dtype=np.float64) - mapping["mean"]) / mapping["std"])
    coded = coded @ mapping["basis"].T
    return to_simplex(mapping["regressor"].predict(coded.reshape(1, -1))[0])


def score_full_delta(run_dir, device="cuda", n_shadows=40, rank=32):
    """Ridge on the full upload-minus-global delta, against ``global_final``.

    Shadows train one round from that checkpoint with the same algorithm, λ,
    and logit adjustment. The saved FedPer head is the global head, so its
    delta is the trunk.
    """
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    device = resolve_device(device or cfg.device)
    histograms, _ = load_labels(run_dir)
    local_states = load_local_states(run_dir)
    global_state = load_global_state(run_dir)

    from fedseismic.experiment import build_federated_run

    data = build_federated_run(cfg, meta.get("seed", cfg.seed))
    images, label_targets, _, _, spec = load_medmnist(cfg.dataset, cfg.data_root, download=False)
    rng = np.random.RandomState(30_000 + int(meta.get("seed", cfg.seed)))
    partitions = [
        indices for indices in partition_dirichlet(label_targets, n_shadows, cfg.partition_alpha, rng)
        if len(indices) >= 2
    ]
    loaders = build_medmnist_loaders(
        images, label_targets, partitions, cfg.batch_size, spec["n_channels"], train=True,
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
        trainer.num_classes = cfg.num_classes
        trainer.logit_adjust = bool(cfg.logit_adjust)
        trainer.download(global_state)
        trainer.train(global_state=global_state)
        features.append(flatten_state_delta(trainer.upload(), global_state))
        targets_pi.append(pi)

    mapping = fit_delta_map(features, targets_pi, rank=rank)
    null_hat = apply_delta_map(np.zeros(features[0].size, dtype=np.float32), mapping)

    leaks = []
    null_leaks = []
    ranks = []
    topks = []
    hats = []
    truths = []
    for client in sorted(set(histograms) & set(local_states)):
        pi = to_simplex(histograms[client])
        hat = apply_delta_map(flatten_state_delta(local_states[client], global_state), mapping)
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
        "logit_adjust": bool(cfg.logit_adjust),
        "seed": meta.get("seed", cfg.seed),
        "n_shadows": len(features),
        "local_mean": utility.get("local_mean"),
        "delta_leakage": attack,
        "delta_baseline": baseline,
        "delta_advantage": attack - baseline,
        "delta_spearman": _mean(ranks),
        "delta_topk": _mean(topks),
        "delta_presence_auc": _presence_auc(hats, truths),
    }
