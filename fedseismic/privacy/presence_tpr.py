"""Per-class presence AUC and true-positive rate at 1% false positives.

Scores are the held-out public softmax of the saved local model, and that
softmax minus the final global model's. Clients are pooled across seeds.
Rare classes are the ones with the lowest frequency in the training set.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from fedseismic.config import RunConfig
from fedseismic.data import load_medmnist
from fedseismic.privacy.estimators import clone_model_from_state
from fedseismic.privacy.fair_leakage import _split_softmax
from fedseismic.privacy.log import load_global_state, load_labels, load_local_states, load_meta
from fedseismic.privacy.metrics import to_simplex


def _tpr_at_fpr(labels, scores, target=0.01):
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target, fpr, tpr))


def collect_run(run_dir, device="cuda"):
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    histograms, _ = load_labels(run_dir)
    local_states = load_local_states(run_dir)
    global_state = load_global_state(run_dir)
    split_at = 0 if cfg.privacy_probe_batches is None else int(cfg.privacy_probe_batches)

    from fedseismic.experiment import build_federated_run

    data = build_federated_run(cfg, meta.get("seed", cfg.seed))
    global_model = clone_model_from_state(data.model_factory, global_state, device=device)
    _, global_late, _, _ = _split_softmax(
        global_model, data.probe_loader, cfg.num_classes, device, split_at,
    )
    rows = []
    for client in sorted(set(histograms) & set(local_states)):
        pi = to_simplex(histograms[client])
        local_model = clone_model_from_state(data.model_factory, local_states[client], device=device)
        _, late, _, _ = _split_softmax(
            local_model, data.probe_loader, cfg.num_classes, device, split_at,
        )
        for class_index in range(cfg.num_classes):
            rows.append({
                "seed": meta.get("seed", cfg.seed),
                "client": client,
                "class_index": class_index,
                "present": int(pi[class_index] > 1e-3),
                "softmax": float(late[class_index]),
                "residual": float(late[class_index] - global_late[class_index]),
            })
    return rows


def summarize(frame, frequencies):
    rows = []
    for (algorithm, lam, class_index), part in frame.groupby(
        ["algorithm", "lambda_cap", "class_index"], dropna=False,
    ):
        labels = part["present"].to_numpy()
        n_neg = int((labels == 0).sum())
        row = {
            "algorithm": algorithm,
            "lambda_cap": lam,
            "class_index": int(class_index),
            "frequency": float(frequencies[int(class_index)]),
            "n": int(len(part)),
            "n_neg": n_neg,
        }
        for name in ("softmax", "residual"):
            scores = part[name].to_numpy()
            if len(np.unique(labels)) < 2:
                row[f"{name}_auc"] = float("nan")
                row[f"{name}_tpr_at_1"] = float("nan")
            else:
                row[f"{name}_auc"] = float(roc_auc_score(labels, scores))
                row[f"{name}_tpr_at_1"] = _tpr_at_fpr(labels, scores)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["algorithm", "lambda_cap", "frequency"])
