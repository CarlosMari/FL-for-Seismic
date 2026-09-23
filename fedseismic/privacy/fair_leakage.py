"""Rescore saved runs without the probe FedKPer already used.

Aggregation in ``infer`` reads the softmax prior on the first
``privacy_probe_batches`` of the public test loader. Fair leakage uses every
later batch. The weight readout is the last-layer bias change from the global
model, not the full local weight vector.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from fedseismic.config import RunConfig, resolve_device
from fedseismic.eval.metrics import _logits
from fedseismic.privacy.estimators import clone_model_from_state, extract_last_linear
from fedseismic.privacy.log import load_global_state, load_labels, load_local_states, load_meta
from fedseismic.privacy.metrics import leakage_from_tv, to_simplex, total_variation


def _split_softmax(model, loader, num_classes, device, split_at):
    """Mean softmax on batches ``[0, split_at)`` and ``[split_at, end)``."""
    model.eval()
    model.to(device)
    early = torch.zeros(num_classes, device=device, dtype=torch.float64)
    late = torch.zeros(num_classes, device=device, dtype=torch.float64)
    early_count = 0
    late_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images = batch[0].to(device, dtype=torch.float)
            probs = F.softmax(_logits(model(images)), dim=1)
            if probs.ndim == 4:
                mean = probs.mean(dim=(0, 2, 3)).double()
            else:
                mean = probs.mean(dim=0).double()
            if batch_index < split_at:
                early = early + mean
                early_count += 1
            else:
                late = late + mean
                late_count += 1
    uniform = np.full(num_classes, 1.0 / max(num_classes, 1), dtype=np.float64)
    early_np = uniform if early_count == 0 else (early / early_count).detach().cpu().numpy()
    late_np = uniform if late_count == 0 else (late / late_count).detach().cpu().numpy()
    return early_np, late_np, early_count, late_count


def _bias_delta(local_state, global_state, num_classes):
    _, local_bias = extract_last_linear(local_state)
    _, global_bias = extract_last_linear(global_state)
    if local_bias is None or global_bias is None:
        return None
    delta = np.asarray(local_bias, dtype=np.float64).reshape(-1) - np.asarray(global_bias, dtype=np.float64).reshape(-1)
    if len(delta) != num_classes:
        return None
    return delta


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


def _mean_leakage(pairs):
    if not pairs:
        return float("nan")
    return float(np.mean([leakage_from_tv(total_variation(true, hat)) for true, hat in pairs]))


def score_run(run_dir, device="cuda"):
    """Leakage of π from a probe the aggregator did not see, plus the bias delta."""
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
    clients = sorted(set(histograms) & set(local_states))
    global_model = clone_model_from_state(data.model_factory, global_state, device=device)
    _, global_late, _, late_count = _split_softmax(
        global_model, probe, cfg.num_classes, device, split_at,
    )

    overlap_pairs = []
    fair_pairs = []
    global_pairs = []
    spearman_pairs = []
    bias_pairs = []
    for client in clients:
        pi = to_simplex(histograms[client])
        local_model = clone_model_from_state(data.model_factory, local_states[client], device=device)
        early, late, _, _ = _split_softmax(local_model, probe, cfg.num_classes, device, split_at)
        overlap_pairs.append((pi, early))
        fair_pairs.append((pi, late))
        global_pairs.append((pi, global_late))
        spearman_pairs.append((pi, late))
        delta = _bias_delta(local_states[client], global_state, cfg.num_classes)
        if delta is not None:
            bias_pairs.append((pi, delta))

    utility = meta.get("utility") or {}
    fair = _mean_leakage(fair_pairs)
    global_leak = _mean_leakage(global_pairs)
    return {
        "run_dir": str(run_dir),
        "algorithm": cfg.algorithm,
        "lambda_cap": None if cfg.algorithm != "fedkper" else cfg.lambda_cap,
        "fedkper_diversity": cfg.fedkper_diversity,
        "seed": meta.get("seed", cfg.seed),
        "local_mean": utility.get("local_mean"),
        "heldout_batches": late_count,
        "overlap_leakage": _mean_leakage(overlap_pairs),
        "fair_leakage": fair,
        "global_leakage": global_leak,
        "fair_advantage": fair - global_leak,
        "fair_spearman": _mean_spearman(spearman_pairs),
        "bias_delta_spearman": _mean_spearman(bias_pairs),
    }
