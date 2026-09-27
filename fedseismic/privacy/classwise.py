"""Per-class accuracy of the deployed local model and the final global model.

Both are scored on each client's own holdout. Counts are pooled across clients,
so a class the client does not have does not enter that client's rate. FedLC
adds the client's train prior at inference. FedPer is read from the saved
personal head when that file exists; the upload still has the global head.
"""

from pathlib import Path

import numpy as np
import torch

from fedseismic.config import RunConfig, resolve_device
from fedseismic.eval.metrics import _logits
from fedseismic.privacy.estimators import clone_model_from_state
from fedseismic.privacy.log import (
    load_global_state,
    load_labels,
    load_local_states,
    load_meta,
    load_personal_states,
)
from fedseismic.privacy.metrics import to_simplex
from fedseismic.privacy.prior_modes import _macro_f1


def _batches(model, loader, device, adjustment=None):
    """Return concatenated predictions and labels for one holdout."""
    model.eval()
    model.to(device)
    predictions = []
    labels = []
    shift = None if adjustment is None else torch.as_tensor(adjustment, dtype=torch.float32, device=device)
    with torch.no_grad():
        for images, targets, *_ in loader:
            images = images.to(device, dtype=torch.float)
            logits = _logits(model(images))
            if shift is not None:
                logits = logits + shift
            predictions.append(logits.argmax(dim=1).detach().cpu())
            labels.append(targets.detach().cpu().long().view(-1))
    if not predictions:
        return None, None
    return torch.cat(predictions), torch.cat(labels)


def score_run(run_dir, device="cpu"):
    """Class rows plus macro-F1 on classes present in the client's train mix."""
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    device = resolve_device(device or cfg.device)
    personal = load_personal_states(run_dir)
    states = personal or load_local_states(run_dir)
    histograms, _ = load_labels(run_dir)
    global_state = load_global_state(run_dir)
    use_prior = bool(getattr(cfg, "logit_adjust", False))

    from fedseismic.experiment import build_federated_run

    data = build_federated_run(cfg, meta.get("seed", cfg.seed))
    global_model = clone_model_from_state(data.model_factory, global_state, device=device)
    local_correct = np.zeros(cfg.num_classes, dtype=np.float64)
    local_total = np.zeros(cfg.num_classes, dtype=np.float64)
    global_correct = np.zeros(cfg.num_classes, dtype=np.float64)
    local_f1 = []
    global_f1 = []
    local_hit = local_seen = 0
    global_hit = global_seen = 0
    for client, state in states.items():
        holdout = data.client_tests[client]
        if holdout is None or not len(holdout.dataset):
            continue
        adjustment = None
        if use_prior and client in histograms:
            adjustment = np.log(np.clip(to_simplex(histograms[client]), 1e-8, None))
        local_model = clone_model_from_state(data.model_factory, state, device=device)
        local_pred, labels = _batches(local_model, holdout, device, adjustment)
        global_pred, _ = _batches(global_model, holdout, device)
        if local_pred is None or global_pred is None:
            continue
        labels_np = labels.numpy()
        local_np = local_pred.numpy()
        global_np = global_pred.numpy()
        for class_index in range(cfg.num_classes):
            mask = labels_np == class_index
            count = int(mask.sum())
            local_total[class_index] += count
            local_correct[class_index] += int((local_np[mask] == class_index).sum())
            global_correct[class_index] += int((global_np[mask] == class_index).sum())
        local_hit += int((local_np == labels_np).sum())
        global_hit += int((global_np == labels_np).sum())
        local_seen += int(labels_np.size)
        global_seen += int(labels_np.size)
        if client in histograms:
            present = to_simplex(histograms[client]) > 1e-3
            local_f1.append(_macro_f1(local_np, labels_np, present))
            global_f1.append(_macro_f1(global_np, labels_np, present))
    rows = []
    for class_index in range(cfg.num_classes):
        denom = local_total[class_index]
        rows.append({
            "class_index": class_index,
            "n": int(denom),
            "local_acc": float(local_correct[class_index] / denom) if denom else float("nan"),
            "global_acc": float(global_correct[class_index] / denom) if denom else float("nan"),
        })
    finite = [value for value in local_f1 if value == value]
    finite_global = [value for value in global_f1 if value == value]
    return {
        "classes": rows,
        "deployed": "personal" if personal else "local",
        "logit_adjust": use_prior,
        "local_acc": float(local_hit / local_seen) if local_seen else float("nan"),
        "global_acc": float(global_hit / global_seen) if global_seen else float("nan"),
        "local_macro_f1": float(np.mean(finite)) if finite else float("nan"),
        "global_macro_f1": float(np.mean(finite_global)) if finite_global else float("nan"),
    }
