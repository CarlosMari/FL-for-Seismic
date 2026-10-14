"""A class-balanced step does not grow when a class is duplicated."""

import unittest
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fedseismic.federated.client import StepClientTrainer


class _Triples(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], int(self.labels[index]), index


def _step(features, labels, balance):
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    start = deepcopy(model.state_dict())
    loader = DataLoader(_Triples(features, labels), batch_size=4, shuffle=False)
    trainer = StepClientTrainer(
        model, loader, nn.CrossEntropyLoss(), device="cpu",
        local_steps=1, gradient_balance=balance, lr=1.0,
        weight_decay=0.0, optimizer="sgd", momentum=0.0, num_classes=2,
    )
    trainer.model.load_state_dict(start)
    trainer.train()
    return {name: tensor.detach().clone() for name, tensor in trainer.model.state_dict().items()}


class LocalStepTests(unittest.TestCase):
    def test_duplicating_a_class_does_not_change_the_balanced_step(self):
        features = torch.tensor([
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [-0.2, 0.9],
        ])
        labels = torch.tensor([0, 0, 1, 1])
        extra = features[:2].repeat(3, 1)
        duplicated = torch.cat([features, extra])
        duplicated_labels = torch.cat([labels, torch.zeros(len(extra), dtype=torch.long)])
        balanced = _step(features, labels, "class")
        balanced_more = _step(duplicated, duplicated_labels, "class")
        weighted = _step(features, labels, "example")
        weighted_more = _step(duplicated, duplicated_labels, "example")
        for name in balanced:
            self.assertTrue(torch.allclose(balanced[name], balanced_more[name], atol=1e-5))
            self.assertFalse(torch.allclose(weighted[name], weighted_more[name], atol=1e-4))


if __name__ == "__main__":
    unittest.main()
