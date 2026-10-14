"""Cohort fingerprint on synthetic residuals: same-cohort pairs score higher."""

import unittest

import numpy as np

from fedseismic.privacy.fingerprint import pairwise_cohort_auc


def _obs(peer, cohort, vector):
    return {"peer": peer, "cohort": tuple(cohort), "bias": np.asarray(vector, dtype=np.float64)}


class FingerprintTests(unittest.TestCase):
    def test_same_cohort_pairs_rank_above(self):
        observations = [
            _obs(0, (0, 1, 2), [1.0, 0.0]),
            _obs(0, (0, 1, 2), [0.9, 0.1]),
            _obs(0, (0, 3, 4), [0.0, 1.0]),
            _obs(0, (0, 3, 4), [0.1, 0.9]),
        ]
        auc, pairs = pairwise_cohort_auc(observations, "bias")
        self.assertEqual(pairs, 6)
        self.assertGreater(auc, 0.9)

    def test_degenerate_single_cohort_is_nan(self):
        observations = [
            _obs(0, (0, 1, 2), [1.0, 0.0]),
            _obs(0, (0, 1, 2), [0.9, 0.1]),
        ]
        auc, _ = pairwise_cohort_auc(observations, "bias")
        self.assertFalse(np.isfinite(auc))

    def test_pairs_never_cross_peers(self):
        observations = [
            _obs(0, (0, 1, 2), [1.0, 0.0]),
            _obs(1, (0, 1, 2), [1.0, 0.0]),
        ]
        _auc, pairs = pairwise_cohort_auc(observations, "bias")
        self.assertEqual(pairs, 0)


if __name__ == "__main__":
    unittest.main()
