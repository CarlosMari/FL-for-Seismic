"""Multi-seed experiment entry point."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .config import (
    CLASSIFICATION_DATASETS, CheckpointPolicy, DatasetName, RunConfig, resolve_device,
)
from .data import (
    MEDMNIST_SPECS,
    build_cifar_loaders,
    build_cifar_test_loader,
    build_client_loaders,
    build_medmnist_loaders,
    build_medmnist_test_loader,
    build_test_loader,
    compute_client_class_info,
    load_and_normalize,
    load_cifar10,
    load_medmnist,
    partition_dirichlet,
    partition_iid,
    partition_iid_indices,
    partition_noniid,
    split_client_local_test,
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
from .models import FedAvgNetCIFAR, MedMNISTNet, UNet


@dataclass
class SeedResults:
    miou_final: list[float]
    miou_best: list[float]
    c5_final: list[float]
    local_mean: list[float]
    local_worst: list[float]
    global_on_local: list[float]
    history: list[list[RoundRecord]]

    @property
    def mean_std(self) -> tuple[float, float]:
        values = np.asarray(self.miou_final, dtype=np.float64)
        return float(values.mean()), float(values.std())

    @property
    def local_mean_std(self) -> tuple[float, float]:
        values = np.asarray(self.local_mean, dtype=np.float64)
        return float(values.mean()), float(values.std())

    @property
    def recovery_rate(self) -> float:
        if not self.c5_final:
            return 0.0
        return float(np.mean(np.asarray(self.c5_final) > 0.15))

    @property
    def balance(self) -> float:
        if not self.miou_final or not self.local_mean:
            return float("nan")
        return 0.5 * (float(np.mean(self.miou_final)) + float(np.mean(self.local_mean)))


def _seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _criterion(cfg, train_labels):
    if cfg.algorithm == "fedkper" or cfg.task == "classification":
        return nn.CrossEntropyLoss()
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


def _loader_kwargs(cfg):
    pin_memory = str(cfg.device).startswith("cuda")
    if cfg.num_workers is not None:
        workers = cfg.num_workers
    else:
        workers = 2 if pin_memory else 0
    return dict(num_workers=workers, pin_memory=pin_memory)


def _required_paths(cfg):
    if cfg.dataset in CLASSIFICATION_DATASETS:
        return
    paths = [cfg.train_seismic, cfg.train_labels]
    if any(path is None for path in paths):
        raise ValueError("train_seismic and train_labels are required for a seismic run")


def _setup_seismic(cfg, seed, rng):
    train_seismic = load_and_normalize(cfg.train_seismic)
    train_labels = np.load(cfg.train_labels)
    partitions = (
        partition_noniid(train_seismic.shape[1], cfg.num_clients)
        if cfg.split == "noniid"
        else partition_iid(train_seismic.shape[1], cfg.num_clients, rng)
    )
    train_parts, test_parts = split_client_local_test(partitions, cfg.local_test_ratio)
    kwargs = _loader_kwargs(cfg)
    loaders = build_client_loaders(
        train_seismic, train_labels, train_parts, cfg.batch_size, **kwargs,
    )
    client_tests = build_client_loaders(
        train_seismic, train_labels, test_parts, cfg.batch_size,
        shuffle=False, train_status=False, **kwargs,
    )
    tests = {}
    for name, seismic, labels in (
        ("test1", cfg.test1_seismic, cfg.test1_labels),
        ("test2", cfg.test2_seismic, cfg.test2_labels),
        ("validation", cfg.validation_seismic, cfg.validation_labels),
    ):
        if seismic is not None and labels is not None:
            tests[name] = build_test_loader(
                seismic, labels, num_workers=kwargs["num_workers"],
                pin_memory=kwargs["pin_memory"],
            )
    model_factory = lambda: UNet(
        1, cfg.num_classes, bilinear=False, norm=cfg.norm, groups=cfg.norm_groups,
    )
    return train_labels, train_parts, loaders, client_tests, tests, model_factory


def _setup_cifar(cfg, seed, rng):
    train_images, train_targets, test_images, test_targets = load_cifar10(cfg.data_root)
    if cfg.num_classes != 10:
        cfg.num_classes = 10
    partitions = (
        partition_dirichlet(train_targets, cfg.num_clients, cfg.partition_alpha, rng)
        if cfg.split == "noniid"
        else partition_iid_indices(len(train_targets), cfg.num_clients, rng)
    )
    train_parts, test_parts = split_client_local_test(
        partitions, cfg.local_test_ratio, rng=rng,
    )
    kwargs = _loader_kwargs(cfg)
    loaders = build_cifar_loaders(
        train_images, train_targets, train_parts, cfg.batch_size, train=True, **kwargs,
    )
    client_tests = build_cifar_loaders(
        train_images, train_targets, test_parts, cfg.batch_size, train=False, **kwargs,
    )
    tests = {
        "test1": build_cifar_test_loader(
            test_images, test_targets, cfg.batch_size, **kwargs,
        ),
    }
    model_factory = lambda: FedAvgNetCIFAR(num_classes=cfg.num_classes, in_channels=3)
    return train_targets, train_parts, loaders, client_tests, tests, model_factory


def _setup_medmnist(cfg, seed, rng):
    train_images, train_targets, test_images, test_targets, spec = load_medmnist(
        cfg.dataset, cfg.data_root,
    )
    cfg.num_classes = spec["n_classes"]
    partitions = (
        partition_dirichlet(train_targets, cfg.num_clients, cfg.partition_alpha, rng)
        if cfg.split == "noniid"
        else partition_iid_indices(len(train_targets), cfg.num_clients, rng)
    )
    train_parts, test_parts = split_client_local_test(
        partitions, cfg.local_test_ratio, rng=rng,
    )
    kwargs = _loader_kwargs(cfg)
    n_channels = spec["n_channels"]
    loaders = build_medmnist_loaders(
        train_images, train_targets, train_parts, cfg.batch_size, n_channels,
        train=True, **kwargs,
    )
    client_tests = build_medmnist_loaders(
        train_images, train_targets, test_parts, cfg.batch_size, n_channels,
        train=False, **kwargs,
    )
    tests = {
        "test1": build_medmnist_test_loader(
            test_images, test_targets, cfg.batch_size, n_channels, **kwargs,
        ),
    }
    model_factory = lambda: MedMNISTNet(
        num_classes=cfg.num_classes, in_channels=n_channels,
    )
    return train_targets, train_parts, loaders, client_tests, tests, model_factory


def _last_or_nan(values):
    measured = [value for value in values if value is not None]
    return float(measured[-1]) if measured else float("nan")


def _run_seed(cfg, seed):
    _required_paths(cfg)
    _seed_everything(seed)
    cfg.device = resolve_device(cfg.device)
    rng = np.random.RandomState(seed)
    if cfg.dataset == DatasetName.CIFAR10.value:
        train_labels, train_parts, loaders, client_tests, tests, model_factory = _setup_cifar(
            cfg, seed, rng,
        )
    elif cfg.dataset in MEDMNIST_SPECS:
        train_labels, train_parts, loaders, client_tests, tests, model_factory = _setup_medmnist(
            cfg, seed, rng,
        )
    else:
        train_labels, train_parts, loaders, client_tests, tests, model_factory = _setup_seismic(
            cfg, seed, rng,
        )
    model = model_factory()
    server = Server(
        model=model, client_loaders=loaders, criterion=_criterion(cfg, train_labels),
        config=cfg, client_info=compute_client_class_info(
            train_labels, train_parts, cfg.num_classes, cfg.rare_classes,
        ), model_factory=model_factory, test_loaders=tests,
        client_test_loaders=client_tests, rng=rng,
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
    return (
        final, best, c5,
        _last_or_nan([record.miou_local_mean for record in history]),
        _last_or_nan([record.miou_local_worst for record in history]),
        _last_or_nan([record.miou_global_on_local for record in history]),
        history,
    )


def run(cfg: RunConfig, seeds=None) -> SeedResults:
    """Run one configuration for every requested seed."""
    seeds = list(seeds if seeds is not None else [cfg.seed])
    results = [_run_seed(cfg, seed) for seed in seeds]
    return SeedResults(
        miou_final=[item[0] for item in results],
        miou_best=[item[1] for item in results],
        c5_final=[item[2] for item in results],
        local_mean=[item[3] for item in results],
        local_worst=[item[4] for item in results],
        global_on_local=[item[5] for item in results],
        history=[item[6] for item in results],
    )
