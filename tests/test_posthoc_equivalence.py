"""Reproduction gate for the published seed ensemble post-hoc metrics."""

from pathlib import Path

import numpy as np
import pytest
import torch

from fedseismic.data import build_test_loader
from fedseismic.eval.posthoc import compute_class_prior, evaluate_ensemble, load_model
from fedseismic.models import UNet


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 123, 7, 99, 2025]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RUN_TEMPLATE = (
    "results/fedavg_noniid_20c_20r_sr0.25_agginvfreq_miou_lossrecall_slice_frc_s{}"
)


def _required_paths():
    paths = [
        ROOT / "datasets/train/train_labels.npy",
        ROOT / "datasets/test_once/test1_seismic.npy",
        ROOT / "datasets/test_once/test1_labels.npy",
        ROOT / "datasets/test_once/test2_seismic.npy",
        ROOT / "datasets/test_once/test2_labels.npy",
    ]
    paths.extend(ROOT / RUN_TEMPLATE.format(seed) / filename
                 for seed in SEEDS for filename in ("global_round_19.pth", "best_global_model.pth"))
    if not all(path.exists() for path in paths):
        pytest.skip("post-hoc checkpoints or seismic datasets are not available")
    return paths


def test_published_ensemble_rows_are_reproduced():
    _required_paths()
    train_labels = np.load(ROOT / "datasets/train/train_labels.npy")
    prior = compute_class_prior(train_labels, num_classes=6)
    log_prior = torch.tensor(np.log(prior + 1e-12), dtype=torch.float32)
    test1_loader, test1_labels = build_test_loader(
        ROOT / "datasets/test_once/test1_seismic.npy",
        ROOT / "datasets/test_once/test1_labels.npy",
        batch_size=4,
    )
    test2_loader, test2_labels = build_test_loader(
        ROOT / "datasets/test_once/test2_seismic.npy",
        ROOT / "datasets/test_once/test2_labels.npy",
        batch_size=4,
    )

    def models(filename):
        return [
            load_model(
                ROOT / RUN_TEMPLATE.format(seed) / filename,
                lambda: UNet(in_channels=1, num_classes=6, bilinear=False),
                device=DEVICE,
            )
            for seed in SEEDS
        ]

    rows = (("global_round_19.pth", 1.5, 0.5965),
            ("best_global_model.pth", 0.5, 0.6155))
    for filename, tau, expected in rows:
        ensemble = models(filename)
        test1 = evaluate_ensemble(ensemble, test1_loader, test1_labels, log_prior, tau)[0]
        test2 = evaluate_ensemble(ensemble, test2_loader, test2_labels, log_prior, tau)[0]
        assert (test1 + test2) / 2.0 == pytest.approx(expected, abs=1e-4)
