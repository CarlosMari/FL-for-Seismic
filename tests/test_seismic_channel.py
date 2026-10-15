"""Seismic channels on synthetic last-round deltas."""

import unittest

import numpy as np

from fedseismic.privacy.seismic_channel import _pearson, _presence_auc


class SeismicChannelTests(unittest.TestCase):
    def test_bias_pearson_tracks_counts(self):
        counts = [np.array([100.0, 1000.0, 500.0]), np.array([200.0, 800.0, 600.0])]
        scores = [np.array([0.01, 0.10, 0.05]), np.array([0.02, 0.08, 0.06])]
        self.assertGreater(_pearson(counts, scores), 0.9)

    def test_presence_separates_absent_classes(self):
        truths = [np.array([0.5, 0.5, 0.0]), np.array([0.0, 0.3, 0.7])]
        scores = [np.array([0.5, 0.4, 0.01]), np.array([0.01, 0.3, 0.6])]
        self.assertGreater(_presence_auc(truths, scores), 0.9)

    def test_degenerate_presence_is_nan(self):
        truths = [np.array([0.5, 0.5]), np.array([0.6, 0.4])]
        scores = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        self.assertFalse(np.isfinite(_presence_auc(truths, scores)))


if __name__ == "__main__":
    unittest.main()
