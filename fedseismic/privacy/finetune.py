"""One round of local training from a saved FedAvg global model.

The fine-tuned weights are the model the client would deploy. They are not
uploaded. Leakage is the held-out softmax of that model minus the final global
softmax.
"""

from pathlib import Path

import numpy as np
import torch.nn as nn

from fedseismic.config import RunConfig, resolve_device
from fedseismic.federated.client import ClientTrainer
from fedseismic.privacy.estimators import clone_model_from_state
from fedseismic.privacy.fair_leakage import _split_softmax
from fedseismic.privacy.log import load_global_state, load_labels, load_meta
from fedseismic.privacy.metrics import leakage_from_tv, to_simplex, total_variation
from fedseismic.federated.server import _accuracy


def score_run(run_dir, device="cuda"):
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    device = resolve_device(device or cfg.device)
    histograms, _ = load_labels(run_dir)
    global_state = load_global_state(run_dir)
    split_at = 0 if cfg.privacy_probe_batches is None else int(cfg.privacy_probe_batches)

    from fedseismic.experiment import build_federated_run

    data = build_federated_run(cfg, meta.get("seed", cfg.seed))
    global_model = clone_model_from_state(data.model_factory, global_state, device=device)
    _, global_late, _, _ = _split_softmax(
        global_model, data.probe_loader, cfg.num_classes, device, split_at,
    )
    accuracies = []
    advantages = []
    for client, loader in enumerate(data.loaders):
        if client not in histograms or len(loader.dataset) < 2:
            continue
        holdout = data.client_tests[client]
        if holdout is None or not len(holdout.dataset):
            continue
        trainer = ClientTrainer(
            data.model_factory(), loader, nn.CrossEntropyLoss(), device=device,
            local_epochs=cfg.local_epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
            optimizer=cfg.optimizer, momentum=cfg.momentum,
        )
        trainer.download(global_state)
        trainer.train(global_state=global_state)
        accuracies.append(_accuracy(trainer.model, holdout, device))
        _, late, _, _ = _split_softmax(
            trainer.model, data.probe_loader, cfg.num_classes, device, split_at,
        )
        pi = to_simplex(histograms[client])
        advantages.append(
            leakage_from_tv(total_variation(pi, late))
            - leakage_from_tv(total_variation(pi, global_late))
        )
    utility = meta.get("utility") or {}
    return {
        "run_dir": str(run_dir),
        "seed": meta.get("seed", cfg.seed),
        "fedavg_local": utility.get("local_mean"),
        "finetune_local": float(np.mean(accuracies)) if accuracies else float("nan"),
        "finetune_advantage": float(np.mean(advantages)) if advantages else float("nan"),
    }
