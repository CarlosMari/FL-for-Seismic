"""The full-delta ridge reads a histogram planted in the weight change."""

import unittest

import numpy as np

from fedseismic.privacy.weight_attack import apply_delta_map, fit_delta_map


class FullDeltaMapTests(unittest.TestCase):
    def test_planted_direction_is_recovered(self):
        rng = np.random.RandomState(0)
        features = []
        targets = []
        for index in range(24):
            pi = np.full(4, 0.05)
            pi[index % 4] = 0.85
            pi = pi / pi.sum()
            feature = rng.normal(0, 0.01, size=16)
            feature[:4] = pi
            features.append(feature)
            targets.append(pi)
        mapping = fit_delta_map(features, targets, rank=4)
        hat = apply_delta_map(features[0], mapping)
        self.assertEqual(int(np.argmax(hat)), int(np.argmax(targets[0])))


if __name__ == "__main__":
    unittest.main()
