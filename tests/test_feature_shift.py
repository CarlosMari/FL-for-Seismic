"""Identical bodies produce a zero public-feature shift."""

import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fedseismic.privacy.feature_shift import public_shift


class _Probe(Dataset):
    def __init__(self):
        self.features = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        self.labels = [0, 1, 0]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, features, get_features=False):
        hidden = self.linear(features)
        if get_features:
            return hidden, hidden
        return hidden


class FeatureShiftTests(unittest.TestCase):
    def test_the_same_body_does_not_move(self):
        torch.manual_seed(0)
        model = _Tiny()
        state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        shift = public_shift(lambda: _Tiny(), state, state, DataLoader(_Probe(), batch_size=3), "cpu", 2)
        self.assertEqual(shift.shape, (2,))
        self.assertTrue(np.allclose(shift, 0.0))


if __name__ == "__main__":
    unittest.main()
