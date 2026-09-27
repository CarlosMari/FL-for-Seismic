"""Public-pad holdout, reweighted loss, and the received-model head delta."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fedseismic.data.partition import drop_indices, holdout_balanced_pad
from fedseismic.federated.client import ClientTrainer
from fedseismic.privacy.log import PrivacyLogger


class _Rows(Dataset):
    def __init__(self, targets):
        self.targets = list(targets)
        self.indices = list(range(len(self.targets)))
        self.images = torch.randn(len(self.targets), 4)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], self.targets[index], index


def _loader(targets):
    return DataLoader(_Rows(targets), batch_size=4, shuffle=False)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(4, 2)

    def forward(self, x):
        return self.classifier(x.view(x.size(0), -1))


class PadTests(unittest.TestCase):
    def test_pad_is_balanced_and_leaves_the_class_in_the_pool(self):
        targets = np.array([0] * 20 + [1] * 20 + [2] * 5)
        pad = holdout_balanced_pad(targets, 4, np.random.RandomState(0))
        self.assertEqual(np.bincount(targets[pad]).tolist(), [4, 4, 4])
        kept, positions = drop_indices(targets, pad)
        self.assertEqual(len(kept), len(targets) - len(pad))
        self.assertTrue(set(pad).isdisjoint(positions.tolist()))
        self.assertGreater(int((kept == 2).sum()), 0)


class DefenseLossTests(unittest.TestCase):
    def test_pad_gives_the_absent_class_a_larger_weight_and_a_private_prior(self):
        trainer = ClientTrainer(
            _Net(), _loader([0] * 8), nn.CrossEntropyLoss(),
            device="cpu", num_classes=2, logit_adjust=True, class_reweight=True,
            public_loader=_loader([0, 0, 1, 1]),
        )
        weight = trainer._loss_weight().numpy()
        self.assertGreater(weight[1], weight[0])
        prior = trainer._log_prior().numpy()
        self.assertAlmostEqual(prior[1], np.log(1e-8), places=4)
        self.assertGreater(prior[0], prior[1])

    def test_public_class_moves_its_row(self):
        torch.manual_seed(0)
        model = _Net()
        nn.init.zeros_(model.classifier.weight)
        nn.init.zeros_(model.classifier.bias)
        trainer = ClientTrainer(
            model, _loader([0] * 8), nn.CrossEntropyLoss(),
            device="cpu", local_epochs=2, lr=0.5, optimizer="sgd", weight_decay=0.0,
            num_classes=2, logit_adjust=True, class_reweight=True,
            public_loader=_loader([1, 1, 1, 1]),
        )
        trainer.train()
        self.assertGreater(float(model.classifier.weight[1].detach().abs().sum()), 0.0)


class HeadDeltaTests(unittest.TestCase):
    def test_head_delta_is_written_for_the_round(self):
        with tempfile.TemporaryDirectory() as folder:
            logger = PrivacyLogger(folder, save_local=False)
            logger.write_head_delta(3, 4, np.ones((2, 3)), np.array([0.25, -0.5]))
            path = Path(folder) / "head" / "round_003_client_004.npz"
            with np.load(path) as handle:
                self.assertEqual(handle["weight_delta"].shape, (2, 3))
                self.assertAlmostEqual(float(handle["bias_delta"][1]), -0.5)


if __name__ == "__main__":
    unittest.main()
