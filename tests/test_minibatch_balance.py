"""Minibatch-balanced steps: every batch holds equal class shares."""

import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fedseismic.federated.client import MinibatchBalancedTrainer


class _Triples(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.targets = labels
        self.indices = list(range(len(labels)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], int(self.labels[index]), index


def _trainer(labels, batch_size=6):
    torch.manual_seed(0)
    np.random.seed(0)
    features = torch.randn(len(labels), 2)
    loader = DataLoader(_Triples(features, labels), batch_size=batch_size, shuffle=False)
    model = nn.Linear(2, 2)
    return MinibatchBalancedTrainer(
        model, loader, nn.CrossEntropyLoss(), device="cpu",
        local_epochs=1, lr=0.01, weight_decay=0.0, optimizer="sgd",
        momentum=0.0, num_classes=2,
    )


class MinibatchBalanceTests(unittest.TestCase):
    def test_every_batch_holds_equal_shares(self):
        labels = torch.tensor([0] * 12 + [1] * 4)
        trainer = _trainer(labels)
        batches = trainer._balanced_batches()
        self.assertTrue(len(batches) > 0)
        for _, targets in batches:
            counts = np.bincount(np.asarray(targets), minlength=2)
            self.assertEqual(counts[0], counts[1])

    def test_epoch_covers_the_largest_class(self):
        labels = torch.tensor([0] * 12 + [1] * 4)
        trainer = _trainer(labels)
        batches = trainer._balanced_batches()
        seen = np.concatenate([np.asarray(targets) for _, targets in batches])
        self.assertEqual(int((seen == 0).sum()), 12)
        self.assertGreaterEqual(int((seen == 1).sum()), 12)

    def test_single_class_client_does_not_crash(self):
        trainer = _trainer(torch.tensor([1] * 7))
        batches = trainer._balanced_batches()
        self.assertTrue(len(batches) > 0)

    def test_max_steps_caps_total_updates(self):
        import torch.nn as nn

        labels = torch.tensor([0] * 40 + [1] * 40)
        torch.manual_seed(0)
        import numpy as np

        np.random.seed(0)
        features = torch.randn(len(labels), 2)
        from torch.utils.data import DataLoader
        loader = DataLoader(_Triples(features, labels), batch_size=8, shuffle=False)
        model = nn.Linear(2, 2)
        steps = {"count": 0}
        real_step = torch.optim.SGD.step

        def counting_step(self):
            steps["count"] += 1
            return real_step(self)

        torch.optim.SGD.step = counting_step
        try:
            trainer = MinibatchBalancedTrainer(
                model, loader, nn.CrossEntropyLoss(), device="cpu",
                local_epochs=5, lr=0.01, weight_decay=0.0, optimizer="sgd",
                momentum=0.0, num_classes=2, max_steps=7,
            )
            trainer.train()
        finally:
            torch.optim.SGD.step = real_step
        self.assertEqual(steps["count"], 7)


if __name__ == "__main__":
    unittest.main()
