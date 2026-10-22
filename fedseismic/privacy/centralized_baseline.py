"""Centralized (no-FL) ceiling for the multi-survey comparison.

One model trained on a whole survey's pooled train set — no federation — and
scored on the same pooled per-site test tails the multi-survey and solo runs
use. It answers "is FL even needed?" as the upper reference next to solo.

Apples-to-apples by construction:
  * the train/test split is the survey's two per-site blocks (``num_clients=2``,
    ``survey_solo``), so the test crosslines are identical to the multi/solo
    runs — only the pooling differs;
  * the loss is the same masked per-survey K=6 softmax ``fedpersurvey`` uses
    (only the survey's own class rows enter the softmax), so head capacity and
    the decision rule match;
  * the budget is ``num_rounds * local_epochs`` passes over the pooled loader,
    which equals the total optimizer steps the federated system performs
    (the pooled loader has as many batches as all clients combined), so
    "FL ~= centralized" means federation costs nothing at matched compute.

Deployed scoring is present-class mIoU with argmax masked to the survey's
class block (identical to the rescore), plus the per-class breakdown.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from fedseismic.config import RunConfig, resolve_device
from fedseismic.experiment import build_federated_run
from fedseismic.privacy.multisurvey_utility import _score_deployed


def _own_classes(loader, num_classes):
    labels = np.concatenate([
        np.asarray(t.detach().cpu() if torch.is_tensor(t) else t).ravel()
        for _, t, _ in loader
    ]).astype(int)
    return torch.as_tensor(np.unique(labels), dtype=torch.long), labels


def _pool(loaders, batch_size, shuffle, seed):
    ds = ConcatDataset([ldr.dataset for ldr in loaders if ldr is not None and len(ldr.dataset)])
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)


def train_centralized(model, train_loader, own, cfg, device, epochs):
    model.to(device)
    opt = (torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                           weight_decay=cfg.weight_decay) if cfg.optimizer == "sgd"
           else torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay))
    own = own.to(device)
    model.train()
    for _ in range(epochs):
        for images, targets, _ in train_loader:
            images = images.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)
            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if own.numel() < logits.shape[1]:  # masked K=6 softmax over own rows
                logits = logits.index_select(1, own)
                targets = torch.searchsorted(own, targets)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def run_centralized(survey, cfg, device, seed=0):
    import dataclasses
    cfg = RunConfig.from_mapping({**dataclasses.asdict(cfg), "survey_solo": survey,
                                  "num_clients": 2})
    data = build_federated_run(cfg, seed)
    block = (0, 6) if str(survey).upper() == "A" else (6, 12)
    train_loader = _pool(data.loaders, cfg.batch_size, True, seed)
    test_loader = _pool(data.client_tests, cfg.batch_size, False, seed)
    own, _ = _own_classes(train_loader, cfg.num_classes)
    epochs = int(cfg.num_rounds) * int(cfg.local_epochs)  # matched total steps
    model = data.model_factory()
    model = train_centralized(model, train_loader, own, cfg, device, epochs)
    miou, per_class, present = _score_deployed(
        model, test_loader, cfg.num_classes, block, device
    )
    return {
        "survey": str(survey).upper(),
        "model": "centralized_noFL",
        "miou": float(miou),
        "per_class": np.asarray(per_class, dtype=np.float64),
        "present": np.asarray(present, dtype=bool),
        "epochs": epochs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/seismic_multisurvey.json")
    ap.add_argument("--surveys", nargs="+", default=["a", "b"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = resolve_device(args.device)
    cfg = RunConfig.from_json(args.base)

    rows, detail = [], []
    for survey in args.surveys:
        res = run_centralized(survey, cfg, device, args.seed)
        print(f"centralized survey {res['survey']}: miou={res['miou']:.4f} "
              f"per-class={np.round(res['per_class'], 3).tolist()}", flush=True)
        rows.append({"survey": res["survey"], "model": res["model"],
                     "seed": args.seed, "miou": res["miou"], "epochs": res["epochs"]})
        for k in range(len(res["per_class"])):
            detail.append({"survey": res["survey"], "client": "pooled", "class": k,
                           "iou": float(res["per_class"][k]), "present": bool(res["present"][k]),
                           "model": res["model"], "seed": args.seed})

    out = Path("results/tables/centralized_baseline.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out, index=False)
    det_out = Path("results/tables/centralized_per_class.csv")
    pd.DataFrame(detail).to_csv(det_out, index=False)
    print(f"wrote {out.resolve()} and {det_out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
