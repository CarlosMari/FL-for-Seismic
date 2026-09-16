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
    local_test_ratio: float = 0.2
    norm: str = "batch"
    norm_groups: int = 8
    checkpoint_policy: CheckpointPolicy = CheckpointPolicy.FINAL
    device: str = "cpu"
    output_dir: str | None = None
    train_seismic: str | None = None
    train_labels: str | None = None
    test1_seismic: str | None = None
    test1_labels: str | None = None
    test2_seismic: str | None = None
    test2_labels: str | None = None
    validation_seismic: str | None = None
    validation_labels: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def task(self) -> str:
        return "classification" if self.dataset == DatasetName.CIFAR10.value else "segmentation"

    def __post_init__(self) -> None:
        self.checkpoint_policy = CheckpointPolicy(self.checkpoint_policy)
        self.dataset = DatasetName(self.dataset).value
        self.rare_classes = tuple(self.rare_classes)
        if self.split not in {"iid", "noniid"}:
            raise ValueError("split must be 'iid' or 'noniid'")
        if self.num_clients < 1 or self.num_rounds < 1 or self.local_epochs < 1:
            raise ValueError("num_clients, num_rounds, and local_epochs must be positive")
        if not 0 < self.sample_ratio <= 1:
            raise ValueError("sample_ratio must be in (0, 1]")
        if not 0 <= self.local_test_ratio < 1:
            raise ValueError("local_test_ratio must be in [0, 1)")
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
