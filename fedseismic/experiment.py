"""Multi-seed experiment entry point."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import torch

from .config import CheckpointPolicy, RunConfig
from .data import (
    build_client_loaders,
    build_test_loader,
    compute_client_class_info,
    load_and_normalize,
    partition_iid,
    partition_noniid,
)
from .federated.server import RoundRecord, Server
from .losses import (
    AsymmetricTverskyLoss,
    CompoundCrosslineRecallLoss,
    FCIoULoss,
    FocalDiceLoss,
    RecallLoss,
    RecallLossPerSlice,
    UnifiedFocalLoss,
)
from .models import UNet


@dataclass
class SeedResults:
    miou_final: list[float]
    miou_best: list[float]
    c5_final: list[float]
    history: list[list[RoundRecord]]

    @property
    def mean_std(self) -> tuple[float, float]:
        values = np.asarray(self.miou_final, dtype=np.float64)
        return float(values.mean()), float(values.std())

    @property
    def recovery_rate(self) -> float:
        if not self.c5_final:
            return 0.0
        return float(np.mean(np.asarray(self.c5_final) > 0.15))


def _seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _criterion(cfg, train_labels):
    if cfg.algorithm == "fedvls":
        from .losses import DiceLoss
        return DiceLoss()
    alpha = None
    if cfg.class_weights:
        _, counts = np.unique(train_labels, return_counts=True)
        frequencies = counts / counts.sum()
        inverse = 1.0 / frequencies
        alpha = (inverse / inverse.sum() * cfg.num_classes).tolist()
    if cfg.loss == "focaldice":
        return FocalDiceLoss(gamma=2.0, alpha=alpha)
    if cfg.loss == "unifiedfocal":
        a = [0.5] * cfg.num_classes
        b = [0.5] * cfg.num_classes
        for class_index in cfg.rare_classes:
            a[class_index], b[class_index] = 0.3, 0.7
        return UnifiedFocalLoss(a, b, gamma_ce=2.0, gamma_t=2.0, lambda_ce=0.5)
    if cfg.loss == "tversky":
        a = [0.5] * cfg.num_classes
        b = [0.5] * cfg.num_classes
        for class_index in cfg.rare_classes:
            a[class_index], b[class_index] = 0.3, 0.7
        return AsymmetricTverskyLoss(a, b, gamma=2.0)
    if cfg.loss == "crossrecall":
        return CompoundCrosslineRecallLoss(cfg.rare_classes, cfg.lambda_crl, alpha=alpha)
    if cfg.loss == "recall":
        return RecallLoss(cfg.num_classes)
    if cfg.loss == "recall_slice":
        return RecallLossPerSlice(cfg.num_classes)
    if cfg.loss == "fciou":
        return FCIoULoss(gamma=2.0)
    raise ValueError(f"unknown loss: {cfg.loss}")


def _required_paths(cfg):
    paths = [cfg.train_seismic, cfg.train_labels]
    if any(path is None for path in paths):
        raise ValueError("train_seismic and train_labels are required for a run")
    return paths


def _run_seed(cfg, seed):
    _required_paths(cfg)
    _seed_everything(seed)
    train_seismic = load_and_normalize(cfg.train_seismic)
    train_labels = np.load(cfg.train_labels)
    rng = np.random.RandomState(seed)
    partitions = (
        partition_noniid(train_seismic.shape[1], cfg.num_clients)
        if cfg.split == "noniid"
        else partition_iid(train_seismic.shape[1], cfg.num_clients, rng)
    )
    loaders = build_client_loaders(
        train_seismic, train_labels, partitions, cfg.batch_size,
    )
    tests = {}
    for name, seismic, labels in (
        ("test1", cfg.test1_seismic, cfg.test1_labels),
        ("test2", cfg.test2_seismic, cfg.test2_labels),
        ("validation", cfg.validation_seismic, cfg.validation_labels),
    ):
        if seismic is not None and labels is not None:
            tests[name] = build_test_loader(seismic, labels)
    model_factory = lambda: UNet(
        1, cfg.num_classes, bilinear=False, norm=cfg.norm, groups=cfg.norm_groups,
    )
    model = model_factory()
    server = Server(
        model=model, client_loaders=loaders, criterion=_criterion(cfg, train_labels),
        config=cfg, client_info=compute_client_class_info(
            train_labels, partitions, cfg.num_classes, cfg.rare_classes,
        ), model_factory=model_factory, test_loaders=tests, rng=rng,
    )
    history = server.run()
    measured = [record.miou_final for record in history if record.miou_final is not None]
    final = float(measured[-1]) if measured else float("nan")
    if cfg.checkpoint_policy is CheckpointPolicy.BEST_TEST:
        best = float(max(measured)) if measured else float("nan")
    elif cfg.checkpoint_policy is CheckpointPolicy.BEST_VAL:
        candidates = [
            (index, record.miou_validation) for index, record in enumerate(history)
            if record.miou_validation is not None and record.miou_final is not None
        ]
        if candidates:
            selected_index = max(candidates, key=lambda item: item[1])[0]
            best = float(history[selected_index].miou_final)
        else:
            best = final
    else:
        best = final
    c5 = float(history[-1].per_class_iou[5]) if history and len(history[-1].per_class_iou) > 5 else 0.0
    return final, best, c5, history


def run(cfg: RunConfig, seeds=None) -> SeedResults:
    """Run one configuration for every requested seed."""
    seeds = list(seeds if seeds is not None else [cfg.seed])
    results = [_run_seed(cfg, seed) for seed in seeds]
    return SeedResults(
        miou_final=[item[0] for item in results],
        miou_best=[item[1] for item in results],
        c5_final=[item[2] for item in results],
        history=[item[3] for item in results],
    )
