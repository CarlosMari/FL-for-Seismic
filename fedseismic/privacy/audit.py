"""Membership with a same-distribution holdout, and presence-F1 diagnostics."""

from pathlib import Path

import numpy as np

from fedseismic.config import RunConfig, resolve_device
from fedseismic.privacy.attacks import evaluate_run
from fedseismic.privacy.estimators import clone_model_from_state, last_layer_prior
from fedseismic.privacy.fair_leakage import _split_softmax
from fedseismic.privacy.log import load_global_state, load_labels, load_local_states, load_meta
from fedseismic.privacy.metrics import to_simplex


def _present_rate(vectors, threshold=1e-3):
    if not vectors:
        return float("nan")
    return float(np.mean([(vector > threshold).mean() for vector in vectors]))


def score_run(run_dir, device="cuda"):
    """Recompute membership and the fraction of classes each attack calls present."""
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
    attack = evaluate_run(run_dir, data=data, device=device)
    clients = sorted(set(histograms) & set(local_states))
    _, global_late, _, _ = _split_softmax(
        clone_model_from_state(data.model_factory, global_state, device=device),
        data.probe_loader, cfg.num_classes, device, split_at,
    )
    softmax_hats = []
    layer_hats = []
    truths = []
    for client in clients:
        local_model = clone_model_from_state(data.model_factory, local_states[client], device=device)
        _, late, _, _ = _split_softmax(local_model, data.probe_loader, cfg.num_classes, device, split_at)
        softmax_hats.append(late)
        layer_hats.append(last_layer_prior(local_states[client], cfg.num_classes))
        truths.append(to_simplex(histograms[client]))
    utility = meta.get("utility") or {}
    return {
        "run_dir": str(run_dir),
        "algorithm": cfg.algorithm,
        "lambda_cap": None if cfg.algorithm != "fedkper" else cfg.lambda_cap,
        "seed": meta.get("seed", cfg.seed),
        "local_mean": utility.get("local_mean"),
        "mia_auc_local": attack["mia_auc_local"],
        "mia_auc_global": attack["mia_auc_global"],
        "true_present_rate": _present_rate(truths),
        "softmax_present_rate": _present_rate(softmax_hats),
        "last_layer_present_rate": _present_rate(layer_hats),
        "global_present_rate": _present_rate([global_late]),
    }
