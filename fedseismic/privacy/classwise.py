"""Per-class accuracy of the saved local model and the final global model.

Both are scored on each client's own holdout. Counts are pooled across clients,
so a class the client does not have does not enter that client's rate.
"""

from pathlib import Path

import numpy as np
import torch

from fedseismic.config import RunConfig, resolve_device
from fedseismic.eval.metrics import _logits
from fedseismic.privacy.estimators import clone_model_from_state
from fedseismic.privacy.log import load_global_state, load_local_states, load_meta


def _counts(model, loader, num_classes, device):
    correct = np.zeros(num_classes, dtype=np.float64)
    total = np.zeros(num_classes, dtype=np.float64)
    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, targets, *_ in loader:
            images = images.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)
            pred = _logits(model(images)).argmax(dim=1)
            for class_index in range(num_classes):
                mask = targets == class_index
                total[class_index] += int(mask.sum().item())
                correct[class_index] += int((pred[mask] == class_index).sum().item())
    return correct, total


def score_run(run_dir, device="cpu"):
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    device = resolve_device(device or cfg.device)
    locals_ = load_local_states(run_dir)
    global_state = load_global_state(run_dir)

    from fedseismic.experiment import build_federated_run

    data = build_federated_run(cfg, meta.get("seed", cfg.seed))
    global_model = clone_model_from_state(data.model_factory, global_state, device=device)
    local_correct = np.zeros(cfg.num_classes, dtype=np.float64)
    local_total = np.zeros(cfg.num_classes, dtype=np.float64)
    global_correct = np.zeros(cfg.num_classes, dtype=np.float64)
    global_total = np.zeros(cfg.num_classes, dtype=np.float64)
    for client, state in locals_.items():
        holdout = data.client_tests[client]
        if holdout is None or not len(holdout.dataset):
            continue
        local_model = clone_model_from_state(data.model_factory, state, device=device)
        correct, total = _counts(local_model, holdout, cfg.num_classes, device)
        local_correct += correct
        local_total += total
        correct, total = _counts(global_model, holdout, cfg.num_classes, device)
        global_correct += correct
        global_total += total
    rows = []
    for class_index in range(cfg.num_classes):
        denom = local_total[class_index]
        rows.append({
            "class_index": class_index,
            "n": int(denom),
            "local_acc": float(local_correct[class_index] / denom) if denom else float("nan"),
            "global_acc": float(global_correct[class_index] / denom) if denom else float("nan"),
        })
    return rows
