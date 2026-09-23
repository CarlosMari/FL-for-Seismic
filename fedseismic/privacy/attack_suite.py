"""Output and classifier-delta attacks on a saved upload.

The global model here is ``global_final.pt``, not the global the client
received. Logit offset and mean softmax use the probe batches after the ones
FedKPer already read. Row norms are ``||W_upload - W_final||`` per class.
"""

from pathlib import Path

import json

import numpy as np

from fedseismic.config import RunConfig, resolve_device
from fedseismic.privacy.estimators import (
    clone_model_from_state,
    extract_last_linear,
    mean_logits,
    softmax_prior,
)
from fedseismic.privacy.log import load_global_state, load_labels, load_local_states, load_meta
from fedseismic.privacy.metrics import leakage_from_tv, presence_mask, to_simplex, total_variation


def offset_prior(local_logits, global_logits):
    """Softmax of the mean logit gap. Recovers π when the gap is log π."""
    gap = np.asarray(local_logits, dtype=np.float64) - np.asarray(global_logits, dtype=np.float64)
    return to_simplex(np.exp(gap - np.max(gap)))


def classifier_row_norms(local_weight, global_weight):
    delta = np.asarray(local_weight, dtype=np.float64) - np.asarray(global_weight, dtype=np.float64)
    return np.linalg.norm(delta.reshape(delta.shape[0], -1), axis=1)


def _advantage(pi, hat, baseline):
    return (
        leakage_from_tv(total_variation(pi, hat))
        - leakage_from_tv(total_variation(pi, baseline))
    )


def _mean(values):
    values = [value for value in values if np.isfinite(value)]
    if not values:
        return float("nan")
    return float(np.mean(values))


def score_run(run_dir, device="cpu"):
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    device = resolve_device(device or "cpu")
    histograms, _ = load_labels(run_dir)
    local_states = load_local_states(run_dir)
    global_state = load_global_state(run_dir)
    start = 0 if cfg.privacy_probe_batches is None else int(cfg.privacy_probe_batches)
    n_batches = getattr(cfg, "privacy_round_batches", None)
    n_batches = None if not n_batches else int(n_batches)

    from fedseismic.experiment import build_federated_run

    data = build_federated_run(cfg, meta.get("seed", cfg.seed))
    probe = data.probe_loader
    global_model = clone_model_from_state(data.model_factory, global_state, device=device)
    global_softmax = softmax_prior(
        global_model, probe, cfg.num_classes, device, max_batches=n_batches, start_batch=start,
    )
    global_logits = mean_logits(
        global_model, probe, cfg.num_classes, device, max_batches=n_batches, start_batch=start,
    )
    global_weight, _ = extract_last_linear(global_state)

    softmax_adv = []
    offset_adv = []
    row_adv = []
    present_norms = []
    absent_norms = []
    for client in sorted(set(histograms) & set(local_states)):
        pi = to_simplex(histograms[client])
        local_model = clone_model_from_state(data.model_factory, local_states[client], device=device)
        local_softmax = softmax_prior(
            local_model, probe, cfg.num_classes, device, max_batches=n_batches, start_batch=start,
        )
        local_logits = mean_logits(
            local_model, probe, cfg.num_classes, device, max_batches=n_batches, start_batch=start,
        )
        softmax_adv.append(_advantage(pi, local_softmax, global_softmax))
        offset_adv.append(_advantage(pi, offset_prior(local_logits, global_logits), global_softmax))
        local_weight, _ = extract_last_linear(local_states[client])
        if local_weight is None or global_weight is None:
            continue
        norms = classifier_row_norms(local_weight, global_weight)
        row_hat = to_simplex(np.exp(norms - np.max(norms)))
        row_adv.append(_advantage(pi, row_hat, global_softmax))
        present = presence_mask(pi)
        if present.any():
            present_norms.append(float(norms[present.astype(bool)].mean()))
        if (~present.astype(bool)).any():
            absent_norms.append(float(norms[~present.astype(bool)].mean()))

    utility = meta.get("utility") or {}
    return {
        "run_dir": str(run_dir),
        "algorithm": cfg.algorithm,
        "seed": meta.get("seed", cfg.seed),
        "local_mean": utility.get("local_mean"),
        "softmax_advantage": _mean(softmax_adv),
        "offset_advantage": _mean(offset_adv),
        "row_advantage": _mean(row_adv),
        "present_row_norm": _mean(present_norms),
        "absent_row_norm": _mean(absent_norms),
        "n_clients": len(softmax_adv),
    }


def bias_advantage(run_dir):
    """Last participation: softmax of the bias delta versus the received global."""
    run_dir = Path(run_dir)
    histograms, _ = load_labels(run_dir)
    last = {}
    with (run_dir / "round_scores.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("bias_delta") is None:
                continue
            last[int(record["client"])] = record
    advantages = []
    for client, record in last.items():
        if client not in histograms:
            continue
        pi = to_simplex(histograms[client])
        delta = np.asarray(record["bias_delta"], dtype=np.float64)
        hat = to_simplex(np.exp(delta - np.max(delta)))
        baseline = np.asarray(record["global_softmax"], dtype=np.float64)
        advantages.append(_advantage(pi, hat, baseline))
    return _mean(advantages)
