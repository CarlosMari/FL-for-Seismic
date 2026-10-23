"""Configuration objects for reproducible seismic FL runs."""

from dataclasses import dataclass, field
from enum import Enum
import json
import warnings
from pathlib import Path
from typing import Any, Mapping

import torch


class CheckpointPolicy(str, Enum):
    FINAL = "final"
    BEST_VAL = "best_val"
    BEST_TEST = "best_test"


class DatasetName(str, Enum):
    SEISMIC = "seismic"
    CIFAR10 = "cifar10"
    BLOODMNIST = "bloodmnist"
    ORGANCMNIST = "organcmnist"
    ORGANSMNIST = "organsmnist"


CLASSIFICATION_DATASETS = {
    DatasetName.CIFAR10.value,
    DatasetName.BLOODMNIST.value,
    DatasetName.ORGANCMNIST.value,
    DatasetName.ORGANSMNIST.value,
}


def resolve_device(device: str) -> str:
    """Honor the configured device; do not silently fall back."""
    requested = device.strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("device='mps' in config but MPS is not available")
        return "mps"
    if requested in {"cuda", "gpu"} or requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"device={device!r} in config but CUDA is not available")
        return "cuda" if requested == "gpu" else requested
    raise ValueError(f"unknown device {device!r}; use cpu, mps, cuda, or cuda:N")


@dataclass
class RunConfig:
    """Knobs for a seismic or CIFAR federated run.

    ``device`` is chosen in config (``mps`` on a Mac, ``cuda`` on a 3090).
    Paths are optional so partitioning and aggregation can be used without data.
    """

    dataset: str = DatasetName.SEISMIC.value
    data_root: str = "datasets"
    partition_alpha: float = 0.1
    num_workers: int | None = None
    num_clients: int = 20
    split: str = "noniid"
    num_rounds: int = 20
    local_epochs: int = 3
    local_steps: int | None = None
    max_steps: int | None = None
    gradient_balance: str = "example"
    infer_log_prior: bool = False
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_classes: int = 6
    sample_ratio: float = 1.0
    seed: int = 42
    loss: str = "focaldice"
    class_weights: bool = False
    agg_strategy: str = "equal"
    force_rare_client: bool = False
    lambda_crl: float = 1.0
    rare_classes: tuple[int, ...] = (4, 5)
    algorithm: str = "fedavg"
    mu: float = 0.01
    lam: float = 0.1
    alpha_proto: float = 0.5
    alpha_dis: float = 0.1
    proto_layer: str = "up4"
    lambda_cap: float = 10.0
    grad_clip: float = 5.0
    fedkper_diversity: str = "infer"
    local_test_ratio: float = 0.2
    spatial_holdout_buffer: int = 0
    balanced_steps: int | None = None
    privacy_delta_dim: int | None = 4096
    privacy_probe_batches: int | None = 8
    privacy_log_last_only: bool = False
    privacy_save_local: bool = True
    privacy_save_received: bool = False
    privacy_round_batches: int | None = None
    ditto_lambda: float = 1.0
    logit_adjust: bool = False
    prior_smooth: float = 0.0
    class_reweight: bool = False
    public_pad_per_class: int = 0
    train_public_pad: bool = False
    presence_rehearsal: bool = False
    dp_clip_norm: float = 5.0
    dp_noise_multiplier: float = 1.0
    norm: str = "batch"
    norm_groups: int = 8
    checkpoint_policy: CheckpointPolicy = CheckpointPolicy.FINAL
    optimizer: str = "adamw"
    momentum: float = 0.9
    device: str = "cpu"
    output_dir: str | None = None
    train_seismic: str | None = None
    train_labels: str | None = None
    test1_seismic: str | None = None
    test1_labels: str | None = None
    multisurvey: bool = False
    survey_b_seismic: str | None = None
    survey_b_labels: str | None = None
    survey_b_offset: int = 6
    survey_c_seismic: str | None = None
    survey_c_labels: str | None = None
    survey_c_offset: int = 12
    survey_solo: str | None = None
    test2_seismic: str | None = None
    test2_labels: str | None = None
    validation_seismic: str | None = None
    validation_labels: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def task(self) -> str:
        return "classification" if self.dataset in CLASSIFICATION_DATASETS else "segmentation"

    def __post_init__(self) -> None:
        self.checkpoint_policy = CheckpointPolicy(self.checkpoint_policy)
        self.dataset = DatasetName(self.dataset).value
        self.rare_classes = tuple(self.rare_classes)
        if self.split not in {"iid", "noniid"}:
            raise ValueError("split must be 'iid' or 'noniid'")
        if self.num_clients < 1 or self.num_rounds < 1 or self.local_epochs < 1:
            raise ValueError("num_clients, num_rounds, and local_epochs must be positive")
        if self.local_steps is not None and int(self.local_steps) < 1:
            raise ValueError("local_steps must be positive")
        if self.max_steps is not None and int(self.max_steps) < 1:
            raise ValueError("max_steps must be positive")
        self.gradient_balance = str(self.gradient_balance).strip().lower()
        if self.gradient_balance not in {"example", "class", "class_minibatch"}:
            raise ValueError("gradient_balance must be 'example', 'class', or 'class_minibatch'")
        if not 0 < self.sample_ratio <= 1:
            raise ValueError("sample_ratio must be in (0, 1]")
        if not 0 <= self.local_test_ratio < 1:
            raise ValueError("local_test_ratio must be in [0, 1)")
        if int(self.spatial_holdout_buffer) < 0:
            raise ValueError("spatial_holdout_buffer must be non-negative")
        if self.balanced_steps is not None and int(self.balanced_steps) < 1:
            raise ValueError("balanced_steps must be positive")
        self.optimizer = str(self.optimizer).strip().lower()
        if self.optimizer not in {"adamw", "sgd"}:
            raise ValueError("optimizer must be 'adamw' or 'sgd'")
        self.fedkper_diversity = str(self.fedkper_diversity).strip().lower()
        if self.fedkper_diversity not in {"oracle", "upload", "infer"}:
            raise ValueError("fedkper_diversity must be 'oracle', 'upload', or 'infer'")
        if self.privacy_delta_dim is not None and self.privacy_delta_dim < 1:
            raise ValueError("privacy_delta_dim must be positive or null")
        if self.public_pad_per_class < 0:
            raise ValueError("public_pad_per_class must be non-negative")
        if not 0.0 <= float(self.prior_smooth) <= 1.0:
            raise ValueError("prior_smooth must be in [0, 1]")
        if self.checkpoint_policy is CheckpointPolicy.BEST_TEST:
            warnings.warn(
                "BEST_TEST selects on the evaluation set and leaks test data",
                UserWarning,
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RunConfig":
        known = {field for field in cls.__dataclass_fields__ if field != "extra"}
        data = {key: value for key, value in values.items() if key in known}
        data["extra"] = {key: value for key, value in values.items() if key not in known}
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> "RunConfig":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        payload = asdict(self)
        payload["checkpoint_policy"] = self.checkpoint_policy.value
        payload["rare_classes"] = list(self.rare_classes)
        extra = payload.pop("extra", {}) or {}
        payload.update(extra)
        return payload
