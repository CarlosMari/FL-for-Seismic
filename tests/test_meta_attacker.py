"""Distinguishing-game scores do not need a trained model."""

import unittest

import numpy as np
import torch.nn as nn

from fedseismic.config import RunConfig
from fedseismic.federated.client import ClientTrainer
from fedseismic.federated.fedkper import FedKPerClientTrainer
from fedseismic.privacy.meta_attacker import make_shadow_trainer, share_accuracy, tpr_at_fpr


class GameTests(unittest.TestCase):
    def test_perfect_ranking_is_caught_at_a_low_false_alarm_rate(self):
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0])
        self.assertGreater(tpr_at_fpr(labels, scores, fpr_cap=0.1), 0.0)
        self.assertLess(tpr_at_fpr(labels, -scores, fpr_cap=0.1), 0.5)

    def test_share_game_reads_a_feature_that_matches_the_high_group(self):
        rng = np.random.RandomState(0)
        shadow = np.zeros((40, 2))
        shadow[:20, 0] = rng.normal(0.0, 0.1, size=20)
        shadow[20:, 0] = rng.normal(2.0, 0.1, size=20)
        mixes = np.zeros((40, 1))
        mixes[20:, 0] = 0.5
        real = np.zeros((8, 2))
        real[:4, 0] = 0.0
        real[4:, 0] = 2.0
        real_mixes = np.zeros((8, 1))
        real_mixes[4:, 0] = 0.5
        self.assertGreater(
            share_accuracy(shadow, mixes, real, real_mixes),
            0.9,
        )

    def test_fedkper_shadows_use_the_distillation_trainer(self):
        model = nn.Linear(4, 8)
        fedkper = make_shadow_trainer(
            RunConfig(algorithm="fedkper", dataset="bloodmnist", num_classes=8, device="cpu"),
            model, object(), "cpu",
        )
        plain = make_shadow_trainer(
            RunConfig(algorithm="fedavg", dataset="bloodmnist", num_classes=8, device="cpu"),
            nn.Linear(4, 8), object(), "cpu",
        )
        self.assertIsInstance(fedkper, FedKPerClientTrainer)
        self.assertIsInstance(plain, ClientTrainer)
        self.assertNotIsInstance(plain, FedKPerClientTrainer)


if __name__ == "__main__":
    unittest.main()
