"""Equivalence gates against the still-present legacy training script."""

import importlib.util
from pathlib import Path
import re

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fedseismic.eval.metrics import evaluate
from fedseismic.data.seismic import build_test_loader
from fedseismic.federated.aggregation import get_agg_weights
from fedseismic.losses import (
    AsymmetricTverskyLoss,
    CompoundCrosslineRecallLoss,
    CrosslineRecallLoss,
    DiceLoss,
    FCIoULoss,
    FocalDiceLoss,
    FocalLoss,
    RecallLoss,
    RecallLossPerSlice,
    UnifiedFocalLoss,
)


ROOT = Path(__file__).resolve().parents[1]
_LEGACY_CACHE = {}


def _legacy(name):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} already retired")
    if name not in _LEGACY_CACHE:
        spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _LEGACY_CACHE[name] = mod
    return _LEGACY_CACHE[name]


def test_all_losses_are_bit_exact_against_legacy():
    old = _legacy("train_federated.py")
    torch.manual_seed(0)
    logits = torch.randn(3, 6, 32, 32)
    targets = torch.randint(0, 6, (3, 32, 32))
    pairs = [
        (old.FocalLoss(), FocalLoss()),
        (old.DiceLoss(), DiceLoss()),
        (old.FocalDiceLoss(), FocalDiceLoss()),
        (old.AsymmetricTverskyLoss([0.5] * 6, [0.5] * 6),
         AsymmetricTverskyLoss([0.5] * 6, [0.5] * 6)),
        (old.UnifiedFocalLoss([0.5] * 6, [0.5] * 6),
         UnifiedFocalLoss([0.5] * 6, [0.5] * 6)),
        (old.CrosslineRecallLoss([4, 5]), CrosslineRecallLoss([4, 5])),
        (old.CompoundCrosslineRecallLoss([4, 5]),
         CompoundCrosslineRecallLoss([4, 5])),
        (old.RecallLoss(6), RecallLoss(6)),
        (old.RecallLossPerSlice(6), RecallLossPerSlice(6)),
        (old.FCIoULoss(), FCIoULoss()),
    ]
    for legacy_loss, migrated_loss in pairs:
        assert float(legacy_loss(logits, targets)) == float(migrated_loss(logits, targets))


class _EncodedSegmentationModel(torch.nn.Module):
    def forward(self, images):
        classes = torch.arange(6, dtype=images.dtype, device=images.device).view(1, 6, 1, 1)
        return -(images - classes).square(), None


def test_all_aggregators_are_bit_exact_against_legacy():
    old = _legacy("train_federated.py")
    info = [
        {
            "num_classes": 5,
            "has_classes": {0, 1, 2, 3, 4},
            "rare_fraction": 0.05,
            "class_fracs": np.array([0.4, 0.2, 0.25, 0.1, 0.05, 0.0]),
        },
        {
            "num_classes": 6,
            "has_classes": {0, 1, 2, 3, 4, 5},
            "rare_fraction": 0.2,
            "class_fracs": np.array([0.2, 0.1, 0.4, 0.1, 0.05, 0.15]),
        },
    ]
    selected = [0, 1]
    class_ious = [
        np.array([0.7, 0.8, 0.9, 0.6, 0.2, 0.0]),
        np.array([0.8, 0.7, 0.9, 0.5, 0.4, 0.3]),
    ]
    images = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1).repeat(2, 1, 1, 1)
    targets = torch.arange(6).view(1, 6, 1).repeat(2, 1, 1)
    loader = DataLoader(TensorDataset(images, targets, torch.arange(2)), batch_size=1)
    model = _EncodedSegmentationModel()
    kwargs = {
        "client_class_ious": class_ious,
        "client_models": [model, model],
        "test_loader": loader,
        "test_labels": np.array([[[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]]),
        "device": "cpu",
    }
    for strategy in (
        "equal", "diversity", "rare", "rare_miou", "invfreq_miou",
        "invfreq", "invfreq_invmiou", "accuracy",
    ):
        legacy_weights = old.get_agg_weights(strategy, selected, info, **kwargs)
        migrated_weights = get_agg_weights(strategy, selected, info, **kwargs)
        assert legacy_weights == migrated_weights, strategy


def test_evaluation_matches_recorded_round_twenty_checkpoint():
    checkpoint = ROOT / (
        "results/fedavg_noniid_20c_20r_sr0.25_agginvfreq_miou_lossrecall_slice_frc_s42"
        "/global_round_19.pth"
    )
    if not checkpoint.exists():
        pytest.skip("round-20 checkpoint is not available")

    data_root = ROOT / "datasets"
    required = [
        data_root / "test_once/test1_seismic.npy",
        data_root / "test_once/test1_labels.npy",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("seismic evaluation dataset is not available")

    log = ROOT / "logs_phase_v/V4_v3_frc_s42.log"
    if not log.exists():
        pytest.skip("matching seed-42 round log is not available")

    from fedseismic.models import UNet

    model = UNet(in_channels=1, num_classes=6, bilinear=False)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    test1_loader, test1_labels = build_test_loader(
        data_root / "test_once/test1_seismic.npy",
        data_root / "test_once/test1_labels.npy",
    )
    test2_loader, test2_labels = build_test_loader(
        data_root / "test_once/test2_seismic.npy",
        data_root / "test_once/test2_labels.npy",
    )
    test1_miou, _ = evaluate(model, test1_loader, test1_labels)
    test2_miou, _ = evaluate(model, test2_loader, test2_labels)
    actual = (test1_miou + test2_miou) / 2.0
    matches = re.findall(
        r"Test1 mIoU:\s+([0-9.]+).*Test2 mIoU:\s+([0-9.]+).*Avg mIoU:\s+([0-9.]+)",
        log.read_text(encoding="utf-8"),
    )
    assert matches, "round metrics not found in matching log"
    expected = float(matches[-1][2])
    assert round(actual, 4) == round(expected, 4)
