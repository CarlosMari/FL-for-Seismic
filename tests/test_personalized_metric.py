"""End-of-run local scoring uses the task metric, not pixel accuracy."""

import unittest
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fedseismic.config import RunConfig
from fedseismic.eval.metrics import score_loader
from fedseismic.federated.server import Server


class _Seg(Dataset):
    """Imbalanced 8x8 segmentation toy: 90% class 0, 10% class 1."""

    def __init__(self, n):
        rng = np.random.RandomState(0)
        self.images = torch.from_numpy(rng.randn(n, 1, 8, 8).astype(np.float32))
        labels = np.zeros((n, 8, 8), dtype=np.int64)
        labels[:, :2, :4] = 1
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index], index


class _Majority(nn.Module):
    """Always predicts class 0: pixel accuracy 0.9, mIoU ~0.45."""

    def forward(self, x):
        n = x.shape[0]
        out = torch.zeros(n, 2, 8, 8)
        out[:, 0] = 1.0
        return out


class PersonalizedMetricTests(unittest.TestCase):
    def test_segmentation_local_score_is_miou_not_pixel_accuracy(self):
        loader = DataLoader(_Seg(4), batch_size=4)
        cfg = RunConfig(num_classes=2)
        server = Server(
            model=_Majority(), client_loaders=[loader], criterion=nn.CrossEntropyLoss(),
            config=cfg, client_test_loaders=[loader], device="cpu",
        )
        trainer = server._trainer(0)
        trainer.model = _Majority()
        mean, _worst = server.evaluate_personalized_local()
        expected = score_loader(trainer.model, loader, 2, "cpu", cfg.task)[0]
        self.assertAlmostEqual(mean, expected, places=5)
        # Pixel accuracy of the majority model is 0.9; mIoU is far lower.
        self.assertLess(mean, 0.7)


if __name__ == "__main__":
    unittest.main()
