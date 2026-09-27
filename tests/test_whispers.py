"""Whispers-style inference on a client whose mix is known."""

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fedseismic.privacy.whispers import (
    TemporalInference,
    advantage_against_mean,
    class_accuracy,
    predict_mixes,
    train_inference,
)


class _Images(Dataset):
    def __init__(self):
        self.images = torch.zeros(6, 1, 2, 2)
        self.targets = torch.tensor([0, 0, 1, 1, 2, 2])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], self.targets[index], index


class _Always(torch.nn.Module):
    def __init__(self, predicted):
        super().__init__()
        self.predicted = predicted

    def forward(self, images):
        return torch.full((images.shape[0], 3), -10.0).index_fill_(1, torch.tensor([self.predicted]), 10.0)


def _one_hot(index, width=4):
    row = np.zeros(width, dtype=np.float64)
    row[index] = 1.0
    return row


class WhispersTest(unittest.TestCase):
    def test_class_accuracy_counts_a_known_prediction(self):
        loader = DataLoader(_Images(), batch_size=2)
        accuracy = class_accuracy(_Always(1), loader, num_classes=3, device="cpu")
        self.assertAlmostEqual(accuracy[1], 1.0)
        self.assertAlmostEqual(accuracy[0], 0.0)

    def test_lstm_recovers_a_mix_written_into_the_matrix(self):
        mixes = [_one_hot(index) for index in range(4)]
        matrices = [[row, row, row] for row in mixes]
        model = train_inference(matrices, mixes, epochs=200, hidden=16, seed=0)
        self.assertIsInstance(model, TemporalInference)
        predicted = predict_mixes(model, matrices)
        for truth, hat in zip(mixes, predicted):
            self.assertEqual(int(hat.argmax()), int(truth.argmax()))
        baseline = np.full(4, 0.25)
        self.assertGreater(advantage_against_mean(mixes, predicted, baseline), 0.5)


if __name__ == "__main__":
    unittest.main()
