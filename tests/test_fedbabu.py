"""FedBABU uploads a body trained under a frozen classifier."""

import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fedseismic.federated.fedbabu import FedBABUClientTrainer


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

    def forward(self, x):
        return self.classifier(torch.relu(self.encoder(x)))


class FedBABUTests(unittest.TestCase):
    def test_upload_keeps_the_initial_head_and_personalize_moves_it(self):
        torch.manual_seed(0)
        model = _Net()
        initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
        trainer = FedBABUClientTrainer(
            model, DataLoader(_Rows(), batch_size=4), nn.CrossEntropyLoss(),
            device="cpu", local_epochs=1, lr=0.2, optimizer="sgd", momentum=0.0,
        )
        trainer.download(initial)
        trainer.train(global_state=initial)
        uploaded = trainer.upload()
        self.assertTrue(torch.equal(uploaded["classifier.weight"], initial["classifier.weight"]))
        self.assertFalse(torch.equal(uploaded["encoder.weight"], initial["encoder.weight"]))
        trainer.personalize()
        self.assertFalse(torch.equal(
            trainer.personal_state["classifier.weight"], initial["classifier.weight"],
        ))
        self.assertTrue(torch.equal(
            trainer.personal_state["encoder.weight"], uploaded["encoder.weight"],
        ))


if __name__ == "__main__":
    unittest.main()
