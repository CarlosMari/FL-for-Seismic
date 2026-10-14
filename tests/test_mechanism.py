"""Slope, bootstrap, and share-bin definitions used by the mechanism checks."""

import unittest

import numpy as np

from fedseismic.privacy.mechanism import (
    band_auc,
    bootstrap_pooled,
    linear_fit,
    pooled_slope,
    prior_presence_auc,
)
from fedseismic.privacy.meta_attacker import in_share_bin, share_bin_metrics


def _planted(n_clients, slope, noise, seed):
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n_clients):
        pi = rng.dirichlet(np.ones(4))
        bias = slope * pi - slope / 4.0 + rng.normal(0.0, noise, size=4)
        rows.append({"client": len(rows), "pi": pi, "bias": bias, "row": np.abs(bias)})
    return rows


class SlopeTests(unittest.TestCase):
    def test_slope_recovers_a_planted_line(self):
        rows = _planted(40, 0.4, 0.005, seed=0)
        self.assertAlmostEqual(pooled_slope(rows), 0.4, delta=0.05)

    def test_within_seed_bootstrap_keeps_both_seeds(self):
        groups = [
            _planted(30, 0.0, 0.002, seed=1),
            _planted(30, 2.0, 0.002, seed=2),
        ]
        point, low, high, _ = bootstrap_pooled(groups, pooled_slope, n_draws=200)
        self.assertGreater(point, 0.7)
        self.assertLess(point, 1.3)
        self.assertGreater(low, 0.5)
        self.assertLess(high, 1.5)

    def test_line_of_slopes(self):
        slope, intercept, r2 = linear_fit([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
        self.assertAlmostEqual(slope, 1.0)
        self.assertAlmostEqual(intercept, 0.0)
        self.assertAlmostEqual(r2, 1.0)


class FloorTests(unittest.TestCase):
    def test_population_score_ranks_the_always_present_class(self):
        rows = []
        for index in range(10):
            pi = np.array([1.0, 0.0]) if index < 5 else np.array([0.5, 0.5])
            rows.append({"pi": pi, "bias": pi, "row": pi})
        self.assertAlmostEqual(prior_presence_auc(rows), 0.833, delta=0.02)


class ShareBinTests(unittest.TestCase):
    def test_edges_follow_the_named_centres(self):
        self.assertTrue(in_share_bin(0.01, 1e-3, 0.03))
        self.assertFalse(in_share_bin(1e-4, 1e-3, 0.03))
        self.assertFalse(in_share_bin(0.20, 0.075, 0.175))
        self.assertFalse(in_share_bin(0.20, 0.25, None))
        self.assertTrue(in_share_bin(0.25, 0.25, None))

    def test_bin_auc_separates_a_planted_score(self):
        rows = []
        for share in (0.0, 0.0, 0.0, 0.01, 0.01, 0.012):
            pi = np.array([share, 1.0 - share])
            rows.append({"pi": pi, "bias": pi, "row": np.array([share, 0.0])})
        self.assertGreater(band_auc(rows, "row", 1e-3, 0.03), 0.9)

    def test_learned_bin_ignores_shares_outside_the_window(self):
        shadow = np.zeros((30, 1))
        mixes = np.zeros((30, 1))
        shadow[:10, 0] = 0.0
        shadow[10:20, 0] = 1.0
        shadow[20:, 0] = 5.0
        mixes[10:20, 0] = 0.05
        mixes[20:, 0] = 0.5
        real = np.array([[0.0], [0.0], [1.0], [1.0]])
        real_mixes = np.array([[0.0], [0.0], [0.05], [0.05]])
        scored = share_bin_metrics(shadow, mixes, real, real_mixes, 0.03, 0.075)
        self.assertGreater(scored["accuracy"], 0.9)
        self.assertGreater(scored["auc"], 0.9)


class PooledBinTests(unittest.TestCase):
    def test_pooled_classifier_reads_the_class_clue(self):
        from fedseismic.privacy.meta_attacker import pooled_bin_metrics

        rng = np.random.RandomState(0)
        shadow = rng.normal(0.0, 0.05, size=(40, 4))
        mixes = np.zeros((40, 2))
        shadow[:20, 0] = 0.0
        shadow[20:, 0] = 2.0
        mixes[20:, 0] = 0.05
        real = np.zeros((8, 4))
        real[:4, 0] = 0.0
        real[4:, 0] = 2.0
        real_mixes = np.zeros((8, 2))
        real_mixes[4:, 0] = 0.05
        scored = pooled_bin_metrics(shadow, mixes, real, real_mixes, 0.03, 0.075)
        self.assertGreater(scored["auc"], 0.9)


if __name__ == "__main__":
    unittest.main()
