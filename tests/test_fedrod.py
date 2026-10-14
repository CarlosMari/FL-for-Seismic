"""FedRoD uploads the generic head and keeps the personal residual local."""

import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fedseismic.federated.fedrod import FedRoDClientTrainer


class _Rows(Dataset):
    def __init__(self):
        self.images = torch.randn(8, 4)
        self.targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], self.targets[index], index


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 3)

    def forward(self, x, get_features=False):
        features = torch.relu(self.encoder(x))
        logits = self.classifier(features)
        if get_features:
            return logits, features
        return logits


class FedRoDTests(unittest.TestCase):
    def test_upload_is_the_generic_head_and_deployed_is_the_sum(self):
        torch.manual_seed(0)
        model = _Net()
        trainer = FedRoDClientTrainer(
            model, DataLoader(_Rows(), batch_size=4), nn.CrossEntropyLoss(),
            device="cpu", local_epochs=2, lr=0.2, num_classes=3,
        )
        trainer.train()
        uploaded = trainer.upload()
        deployed = trainer.personal_state
        self.assertFalse(torch.allclose(trainer.personal_head.weight, torch.zeros_like(trainer.personal_head.weight)))
        self.assertTrue(torch.allclose(uploaded["classifier.weight"], model.classifier.weight))
        self.assertTrue(torch.allclose(
            deployed["classifier.weight"],
            uploaded["classifier.weight"] + trainer.personal_head.weight,
        ))
        self.assertTrue(torch.allclose(uploaded["encoder.weight"], deployed["encoder.weight"]))


if __name__ == "__main__":
    unittest.main()
