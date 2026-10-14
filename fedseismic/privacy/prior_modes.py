"""Which part of the client prior buys the local-accuracy gap.

Three corrections of the same global logits, using the client's true train mix:
the full log-prior, a uniform prior over classes that are present, and the
proportions with absent classes floored instead of suppressed.
"""

from pathlib import Path

import numpy as np
import torch

from fedseismic.config import RunConfig, resolve_device
from fedseismic.eval.metrics import _logits
from fedseismic.privacy.estimators import clone_model_from_state
from fedseismic.privacy.log import load_global_state, load_labels, load_local_states, load_meta
from fedseismic.privacy.metrics import to_simplex


def _adjustment(pi, mode, eps=1e-6):
    pi = np.asarray(pi, dtype=np.float64)
    if mode == "full":
        return np.log(pi + eps)
    if mode == "presence":
        present = (pi > 1e-3).astype(np.float64)
        total = present.sum()
        if total <= 0:
            return np.zeros_like(pi)
        return np.log(present / total + eps)
    if mode == "shape":
        return np.log(np.clip(pi, 1e-2, None))
    raise ValueError(mode)


def _predict(model, loader, device):
    model.eval()
    model.to(device)
    predictions = []
    labels = []
    with torch.no_grad():
        for images, targets, *_ in loader:
            images = images.to(device, dtype=torch.float)
            logits = _logits(model(images)).detach().cpu()
            predictions.append(logits)
            labels.append(targets.detach().cpu().long())
    if not predictions:
        return None, None
    return torch.cat(predictions, dim=0), torch.cat(labels, dim=0)


def _accuracy(logits, labels, adjustment=None):
    scores = logits if adjustment is None else logits + torch.as_tensor(adjustment, dtype=logits.dtype)
    return float((scores.argmax(dim=1) == labels).float().mean().item())


def _macro_f1(predictions, labels, present):
    scores = []
    for class_index in np.flatnonzero(present):
        in_class = labels == class_index
        if int(in_class.sum()) == 0:
            continue
        true_positive = int(((predictions == class_index) & in_class).sum())
        predicted = int((predictions == class_index).sum())
        support = int(in_class.sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _mean(values):
    values = [value for value in values if value == value]
    if not values:
        return float("nan")
    return float(np.mean(values))


def score_run(run_dir, device="cuda"):
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    device = resolve_device(device or cfg.device)
    histograms, _ = load_labels(run_dir)
    local_states = load_local_states(run_dir)
    global_state = load_global_state(run_dir)

    from fedseismic.experiment import build_federated_run

    data = build_federated_run(cfg, meta.get("seed", cfg.seed))
    global_model = clone_model_from_state(data.model_factory, global_state, device=device)
    rows = []
    for client in sorted(set(histograms) & set(local_states)):
        loader = data.client_tests[client]
        if loader is None or not len(loader.dataset):
            continue
        pi = to_simplex(histograms[client])
        present = pi > 1e-3
        global_logits, labels = _predict(global_model, loader, device)
        local_model = clone_model_from_state(data.model_factory, local_states[client], device=device)
        local_logits, _ = _predict(local_model, loader, device)
        if global_logits is None or local_logits is None:
            continue
        local_pred = local_logits.argmax(dim=1)
        global_pred = global_logits.argmax(dim=1)
        rows.append({
            "local": _accuracy(local_logits, labels),
            "global": _accuracy(global_logits, labels),
            "full": _accuracy(global_logits, labels, _adjustment(pi, "full")),
            "presence": _accuracy(global_logits, labels, _adjustment(pi, "presence")),
            "shape": _accuracy(global_logits, labels, _adjustment(pi, "shape")),
            "local_macro_f1": _macro_f1(local_pred.numpy(), labels.numpy(), present),
            "global_macro_f1": _macro_f1(global_pred.numpy(), labels.numpy(), present),
        })

    local = np.asarray([row["local"] for row in rows])
    order = np.argsort(local)
    tail = max(1, int(np.ceil(0.1 * len(order))))
    tail_index = order[:tail]
    global_acc = np.asarray([row["global"] for row in rows])
    full = _mean(row["full"] for row in rows)
    presence = _mean(row["presence"] for row in rows)
    shape = _mean(row["shape"] for row in rows)
    global_mean = _mean(row["global"] for row in rows)
    local_mean = _mean(row["local"] for row in rows)
    full_lift = full - global_mean
    return {
        "run_dir": str(run_dir),
        "algorithm": cfg.algorithm,
        "lambda_cap": None if cfg.algorithm != "fedkper" else cfg.lambda_cap,
        "seed": meta.get("seed", cfg.seed),
        "local_mean": local_mean,
        "global_on_local": global_mean,
        "full": full,
        "presence": presence,
        "shape": shape,
        "presence_of_full": float("nan") if abs(full_lift) < 1e-8 else (presence - global_mean) / full_lift,
        "shape_of_full": float("nan") if abs(full_lift) < 1e-8 else (shape - global_mean) / full_lift,
        "local_macro_f1": _mean(row["local_macro_f1"] for row in rows),
        "global_macro_f1": _mean(row["global_macro_f1"] for row in rows),
        "worst10_local": float(local[tail_index].mean()),
        "worst10_global": float(global_acc[tail_index].mean()),
    }
