"""Distribution inference on a synthetic client whose mix is known."""

import unittest

import numpy as np

from fedseismic.privacy.distribution import (
    entropy_r2,
    histogram_r2,
    leave_client_advantage,
    presence_table,
)


def _one_hot(index, width=4):
    row = np.zeros(width, dtype=np.float64)
    row[index] = 1.0
    return row


class DistributionInferenceTest(unittest.TestCase):
    def test_softmax_names_the_class_and_entropy_does_not(self):
        by_client = {}
        for client in range(4):
            pi = _one_hot(client)
            uniform = np.full(4, 0.25)
            by_client[client] = [(1, pi, pi.copy(), uniform)]
        scored = leave_client_advantage(by_client)
        self.assertGreater(scored["softmax_r2"], 0.9)
        self.assertLess(scored["scalar_r2"], 0.15)
        self.assertGreater(scored["softmax_advantage"], 0.5)
        self.assertLess(scored["scalar_advantage"], 0.15)

    def test_entropy_tracks_skew_when_the_upload_matches(self):
        skewed = [np.array([0.7, 0.1, 0.1, 0.1]), np.array([0.4, 0.2, 0.2, 0.2])]
        flat = [np.full(4, 0.25), np.full(4, 0.25)]
        self.assertGreater(entropy_r2(skewed, skewed), 0.99)
        self.assertLess(entropy_r2(skewed, flat), 0.0)

    def test_presence_residual_ranks_a_known_class(self):
        import pandas as pd

        frame = pd.DataFrame([
            {"class_index": 0, "present": 1, "residual": 0.4},
            {"class_index": 0, "present": 1, "residual": 0.2},
            {"class_index": 0, "present": 0, "residual": -0.1},
            {"class_index": 0, "present": 0, "residual": -0.3},
        ])
        table = presence_table(frame, min_negatives=2)
        self.assertGreater(table.iloc[0]["presence_auc"], 0.9)
        self.assertTrue(bool(table.iloc[0]["reliable"]))

    def test_histogram_r2_is_one_on_a_perfect_copy(self):
        rows = [_one_hot(index) for index in range(3)]
        self.assertAlmostEqual(histogram_r2(rows, rows), 1.0)


if __name__ == "__main__":
    unittest.main()
