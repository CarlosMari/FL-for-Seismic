"""Multi-seed experiment entry point."""

from dataclasses import dataclass
from pathlib import Path
import json

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
    drop_indices,
    holdout_balanced_pad,
    load_and_normalize,
    load_cifar10,
    load_medmnist,
    partition_dirichlet,
    partition_presence,
    partition_iid,
    partition_iid_indices,
    partition_noniid,
    remap_partitions,
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
from .privacy.log import PrivacyLogger


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


@dataclass
class FederatedData:
    train_labels: np.ndarray
    train_parts: list
    loaders: list
    client_tests: list
    tests: dict
    model_factory: object
    client_info: list
    public_loader: object = None

    @property
    def probe_loader(self):
        entry = self.tests.get("test1")
        if entry is None:
            return None
        return entry[0] if isinstance(entry, tuple) else entry


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
    if bool(getattr(cfg, "multisurvey", False)):
        missing = [cfg.survey_b_seismic, cfg.survey_b_labels]
        if any(path is None for path in missing):
            raise ValueError("survey_b_seismic and survey_b_labels are required for a multisurvey run")


def _load_cube_pair(seismic_path, labels_path):
    """Load a survey cube pair. Supports F3-style .npy pairs and the
    Parihaka .npz release (keys ``data``/``labels``, classes 1-6)."""
    if str(seismic_path).endswith(".npz"):
        seismic = np.load(seismic_path)["data"]
        labels = np.load(labels_path)["labels"]
        seismic = (seismic - seismic.min()) / (seismic.max() - seismic.min() + 1e-8)
        return np.asarray(seismic, dtype=np.float64), np.asarray(labels).reshape(seismic.shape).astype(np.uint8)
    return load_and_normalize(seismic_path), np.load(labels_path)


def _setup_seismic(cfg, seed, rng):
    train_seismic = load_and_normalize(cfg.train_seismic)
    train_labels = np.load(cfg.train_labels)
    partitions = (
        partition_noniid(train_seismic.shape[1], cfg.num_clients)
        if cfg.split == "noniid"
        else partition_iid(train_seismic.shape[1], cfg.num_clients, rng)
    )
    train_parts, test_parts = split_client_local_test(
        partitions, cfg.local_test_ratio, buffer=int(getattr(cfg, "spatial_holdout_buffer", 0)))
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
    return train_labels, train_parts, loaders, client_tests, tests, model_factory, None


def _multisurvey_surveys(cfg):
    """(seismic_path, labels_path, offset) for the selected surveys."""
    solo = str(getattr(cfg, "survey_solo", None) or "").strip().lower() or None
    surveys = []
    if solo in (None, "a"):
        surveys.append((cfg.train_seismic, cfg.train_labels, 0))
    if solo in (None, "b"):
        surveys.append((cfg.survey_b_seismic, cfg.survey_b_labels, int(cfg.survey_b_offset)))
    return surveys


def _multisurvey_client_info(cfg):
    """Per-client class info over the 12-class union space (deterministic)."""
    from fedseismic.data.partition import partition_noniid

    surveys = _multisurvey_surveys(cfg)
    per_survey = cfg.num_clients // len(surveys)
    infos = []
    for seismic_path, labels_path, offset in surveys:
        seismic, labels = _load_cube_pair(seismic_path, labels_path)
        labels = np.asarray(labels).astype(int) - (1 if offset else 0) + offset
        chunks = partition_noniid(seismic.shape[1], per_survey)
        tparts, _ = split_client_local_test(
            chunks, cfg.local_test_ratio, buffer=int(getattr(cfg, "spatial_holdout_buffer", 0)))
        infos.extend(compute_client_class_info(labels, tparts, cfg.num_classes, cfg.rare_classes))
    return infos


def _setup_multisurvey(cfg, seed, rng):
    """Setup A: two surveys, contiguous-block sites, one union label space.

    The first half of the clients are survey A (F3, classes 0-5); the second
    half are survey B (Parihaka, remapped to ``survey_b_offset``..+5).
    ``survey_solo`` restricts to one survey ('a' or 'b') for the local-only
    baselines. Normalization is per survey. The test split honors
    ``spatial_holdout_buffer`` like the single-survey path.
    """
    from fedseismic.data.partition import partition_noniid

    if cfg.num_classes != 12:
        cfg.num_classes = 12
    solo = str(getattr(cfg, "survey_solo", None) or "").strip().lower() or None
    if solo not in (None, "a", "b"):
        raise ValueError("survey_solo must be 'a', 'b', or null")
    surveys = _multisurvey_surveys(cfg)
    if cfg.num_clients % len(surveys) != 0:
        raise ValueError("num_clients must split evenly across the selected surveys")
    per_survey = cfg.num_clients // len(surveys)
    kwargs = _loader_kwargs(cfg)
    train_parts, loaders, client_tests = [], [], []
    tests = {}
    for seismic_path, labels_path, offset in surveys:
        if seismic_path is None or labels_path is None:
            raise ValueError("train_seismic/train_labels and survey_b_* paths are required")
        seismic, labels = _load_cube_pair(seismic_path, labels_path)
        labels = (np.asarray(labels).astype(int) - (1 if offset else 0) + offset)
        chunks = partition_noniid(seismic.shape[1], per_survey)
        tparts, eparts = split_client_local_test(
            chunks, cfg.local_test_ratio, buffer=int(getattr(cfg, "spatial_holdout_buffer", 0)))
        train_parts.extend(tparts)
        loaders.extend(build_client_loaders(seismic, labels, tparts, cfg.batch_size, **kwargs))
        client_tests.extend(build_client_loaders(
            seismic, labels, eparts, cfg.batch_size, shuffle=False, train_status=False, **kwargs))
    for name, seismic, labels in (
        ("test1", cfg.test1_seismic, cfg.test1_labels),
        ("test2", cfg.test2_seismic, cfg.test2_labels),
    ):
        if seismic is not None and labels is not None:
            tests[name] = build_test_loader(
                seismic, labels, num_workers=kwargs["num_workers"],
                pin_memory=kwargs["pin_memory"],
            )
    model_factory = lambda: UNet(
        1, cfg.num_classes, bilinear=False, norm=cfg.norm, groups=cfg.norm_groups,
    )
    # train_labels is unused downstream for multisurvey (client_info comes
    # from _multisurvey_client_info; _criterion ignores labels here).
    return None, train_parts, loaders, client_tests, tests, model_factory, None


def _public_pad(cfg, targets, seed):
    """Hold a balanced pad out of the training pool. ``kept`` maps subset indices back."""
    per_class = int(getattr(cfg, "public_pad_per_class", 0) or 0)
    if per_class < 1:
        return targets, None, []
    pad = holdout_balanced_pad(targets, per_class, np.random.RandomState(int(seed) + 17001))
    kept_targets, kept = drop_indices(targets, pad)
    return kept_targets, kept, pad


def _client_partitions(cfg, targets, rng):
    if cfg.presence_rehearsal:
        partitions, rare = partition_presence(targets, cfg.num_clients, rng)
        cfg.rare_classes = tuple(int(class_index) for class_index in rare)
        return partitions
    if cfg.split == "noniid":
        return partition_dirichlet(targets, cfg.num_clients, cfg.partition_alpha, rng)
    return partition_iid_indices(len(targets), cfg.num_clients, rng)


def _train_splits_or_retry(cfg, pool_targets, kept, seed, base_rng, tries=25):
    """Client train/test splits with no empty train client.

    Extreme Dirichlet draws can leave a client with nothing to train on,
    which crashes the train DataLoader. Attempt 0 uses ``base_rng`` exactly
    as before, so published splits are unchanged. Later attempts use derived
    seeds (deterministic given the run seed) and warn. Finished runs never
    triggered this, so their splits are unchanged. ``howmuch.real_counts``
    uses the same helper, so its label check stays in sync.
    """
    import warnings

    for attempt in range(int(tries)):
        rng = base_rng if attempt == 0 else np.random.RandomState(
            int(seed) * 7919 + attempt * 131 + 7)
        partitions = _client_partitions(cfg, pool_targets, rng)
        if kept is not None:
            partitions = remap_partitions(partitions, kept)
        train_parts, test_parts = split_client_local_test(
            partitions, cfg.local_test_ratio, rng=rng,
        )
        empty = [index for index, part in enumerate(train_parts) if len(part) == 0]
        if not empty:
            if attempt > 0:
                warnings.warn(f"partition retry {attempt} for seed {seed}")
            return train_parts, test_parts
    raise ValueError(f"no non-empty client partition in {tries} tries (seed {seed})")


def _setup_cifar(cfg, seed, rng):
    train_images, train_targets, test_images, test_targets = load_cifar10(cfg.data_root)
    if cfg.num_classes != 10:
        cfg.num_classes = 10
    pool_targets, kept, pad = _public_pad(cfg, train_targets, seed)
    train_parts, test_parts = _train_splits_or_retry(
        cfg, pool_targets, kept, seed, rng,
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
    public_loader = None
    if pad:
        public_loader = build_cifar_loaders(
            train_images, train_targets, [pad], cfg.batch_size, train=True, **kwargs,
        )[0]
    return train_targets, train_parts, loaders, client_tests, tests, model_factory, public_loader


def _setup_medmnist(cfg, seed, rng):
    train_images, train_targets, test_images, test_targets, spec = load_medmnist(
        cfg.dataset, cfg.data_root,
    )
    cfg.num_classes = spec["n_classes"]
    pool_targets, kept, pad = _public_pad(cfg, train_targets, seed)
    train_parts, test_parts = _train_splits_or_retry(
        cfg, pool_targets, kept, seed, rng,
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
    public_loader = None
    if pad:
        public_loader = build_medmnist_loaders(
            train_images, train_targets, [pad], cfg.batch_size, n_channels,
            train=True, **kwargs,
        )[0]
    return train_targets, train_parts, loaders, client_tests, tests, model_factory, public_loader


def build_federated_run(cfg: RunConfig, seed):
    """Partition data, build loaders, and return everything a seed run needs."""
    _required_paths(cfg)
    rng = np.random.RandomState(seed)
    multisurvey = bool(getattr(cfg, "multisurvey", False))
    if cfg.dataset == DatasetName.CIFAR10.value:
        train_labels, train_parts, loaders, client_tests, tests, model_factory, public_loader = _setup_cifar(
            cfg, seed, rng,
        )
    elif cfg.dataset in MEDMNIST_SPECS:
        train_labels, train_parts, loaders, client_tests, tests, model_factory, public_loader = _setup_medmnist(
            cfg, seed, rng,
        )
    elif multisurvey:
        train_labels, train_parts, loaders, client_tests, tests, model_factory, public_loader = _setup_multisurvey(
            cfg, seed, rng,
        )
    else:
        train_labels, train_parts, loaders, client_tests, tests, model_factory, public_loader = _setup_seismic(
            cfg, seed, rng,
        )
    if multisurvey:
        client_info = _multisurvey_client_info(cfg)
    else:
        client_info = compute_client_class_info(
            train_labels, train_parts, cfg.num_classes, cfg.rare_classes,
        )
    return FederatedData(
        train_labels=train_labels, train_parts=train_parts, loaders=loaders,
        client_tests=client_tests, tests=tests, model_factory=model_factory,
        client_info=client_info, public_loader=public_loader,
    )


def _privacy_logger(cfg, seed_dir):
    if not cfg.output_dir:
        return None
    return PrivacyLogger(
        seed_dir,
        projection_dim=cfg.privacy_delta_dim,
        projection_seed=cfg.seed,
        log_last_only=cfg.privacy_log_last_only,
        save_local=cfg.privacy_save_local,
    )


def _last_or_nan(values):
    measured = [value for value in values if value is not None]
    return float(measured[-1]) if measured else float("nan")


def _git_head():
    """Current commit hash, or null outside a git checkout. Recorded in
    every run's metadata so results can always be traced to the code that
    produced them (uncommitted-code results are not valid)."""
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=10,
        ).stdout.strip() or None
    except Exception:
        return None


def _git_dirty():
    """True when the working tree has uncommitted changes. A run with a
    hash but a dirty tree is not reproducible from git alone."""
    import subprocess

    try:
        return bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            timeout=10,
        ).stdout.strip())
    except Exception:
        return None


def _run_seed(cfg, seed):
    _seed_everything(seed)
    git_head = _git_head()
    git_dirty = _git_dirty()
    cfg.device = resolve_device(cfg.device)
    data = build_federated_run(cfg, seed)
    seed_dir = None
    logger = None
    if cfg.output_dir:
        seed_dir = Path(cfg.output_dir)
        if seed_dir.name != f"seed_{seed}":
            seed_dir = seed_dir / f"seed_{seed}"
        logger = _privacy_logger(cfg, seed_dir)
        if logger is not None:
            logger.write_ground_truth(
                [info["class_fracs"] for info in data.client_info],
                rare_classes=cfg.rare_classes,
            )
    model = data.model_factory()
    server = Server(
        model=model, client_loaders=data.loaders, criterion=_criterion(cfg, data.train_labels),
        config=cfg, client_info=data.client_info, model_factory=data.model_factory,
        test_loaders=data.tests, client_test_loaders=data.client_tests,
        rng=np.random.RandomState(seed), probe_loader=data.probe_loader,
        privacy_logger=logger, public_loader=data.public_loader,
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
    local_mean = _last_or_nan([record.miou_local_mean for record in history])
    local_worst = _last_or_nan([record.miou_local_worst for record in history])
    global_on_local = _last_or_nan([record.miou_global_on_local for record in history])
    if logger is not None:
        logger.write_meta({
            "seed": seed,
            "config": cfg.to_dict(),
            "git_head": git_head,
            "git_dirty": git_dirty,
            "utility": {
                "local_mean": local_mean,
                "local_worst": local_worst,
                "global_on_local": global_on_local,
                "global_score": final,
            },
        })
        (Path(seed_dir) / "results.json").write_text(json.dumps({
            "final": final, "best": best, "c5": c5,
            "local_mean": local_mean, "local_worst": local_worst,
            "global_on_local": global_on_local,
            "git_head": git_head,
            "git_dirty": git_dirty,
        }, indent=2), encoding="utf-8")
    return (
        final, best, c5, local_mean, local_worst, global_on_local, history,
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
