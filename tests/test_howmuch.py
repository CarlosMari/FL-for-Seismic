"""The how-much scores are zero at their floors and one when the truth is named."""

import unittest

import numpy as np

from fedseismic.privacy.howmuch import (
    _apply_pair,
    _fit_pair,
    best_count_row,
    channels_passing_counts,
    condition_on_support,
    counts_r2,
    proportions_r2,
    renormalize,
    size_r2,
    uniform_on_support,
)


class HowMuchTests(unittest.TestCase):
    def test_per_class_mean_count_scores_zero_and_the_truth_scores_one(self):
        truth = [np.array([2.0, 0.0, 4.0]), np.array([6.0, 2.0, 0.0])]
        mean = np.mean(np.stack(truth), axis=0)
        self.assertAlmostEqual(counts_r2(truth, [mean, mean]), 0.0)
        self.assertAlmostEqual(counts_r2(truth, truth), 1.0)

    def test_mean_dataset_size_scores_zero(self):
        truth = [np.array([1.0, 1.0]), np.array([3.0, 5.0])]
        sizes = [float(np.sum(row)) for row in truth]
        self.assertAlmostEqual(size_r2(truth, [np.mean(sizes), np.mean(sizes)]), 0.0)
        self.assertAlmostEqual(size_r2(truth, sizes), 1.0)

    def test_uniform_on_the_true_support_scores_zero(self):
        pis = [np.array([0.25, 0.75, 0.0]), np.array([0.0, 0.5, 0.5])]
        uniform = [uniform_on_support(pi) for pi in pis]
        truth = [condition_on_support(pi) for pi in pis]
        self.assertAlmostEqual(proportions_r2(pis, uniform), 0.0)
        self.assertAlmostEqual(proportions_r2(pis, truth), 1.0)

    def test_renormalizing_counts_on_the_true_support_recovers_the_share(self):
        pi = np.array([0.25, 0.75, 0.0])
        counts = np.array([10.0, 30.0, 4.0])
        predicted = renormalize(counts, pi > 1e-3)
        self.assertTrue(np.allclose(predicted, condition_on_support(pi)))
        self.assertAlmostEqual(proportions_r2([pi], [predicted]), 1.0)

    def test_best_count_channel_ignores_the_joint_ridge(self):
        table = []
        for dataset in ("Blood", "OrganC", "CIFAR-10"):
            for attacker, estimate, low in (
                ("row", 0.20, 0.10),
                ("decaf", 0.40, 0.20),
                ("bias", 0.05, -0.10),
                ("ridge", 0.90, 0.50),
            ):
                table.append({
                    "dataset": dataset,
                    "method": "FedAvg",
                    "prior_smooth": 1.0,
                    "attacker": attacker,
                    "metric": "counts_r2",
                    "estimate": estimate,
                    "low": low,
                    "floor": 0.0,
                })
        table.append({
            "dataset": "Blood",
            "method": "FedLC",
            "prior_smooth": 0.0,
            "attacker": "row",
            "metric": "counts_r2",
            "estimate": 0.30,
            "low": 0.10,
            "floor": 0.0,
        })
        table.append({
            "dataset": "Blood",
            "method": "FedLC",
            "prior_smooth": 0.0,
            "attacker": "decaf",
            "metric": "counts_r2",
            "estimate": 0.12,
            "low": 0.04,
            "floor": 0.0,
        })
        self.assertEqual(channels_passing_counts(table), ["row", "decaf"])
        fedavg = best_count_row(table, "Blood", "FedAvg", 1.0)
        fedlc = best_count_row(table, "Blood", "FedLC", 0.0)
        self.assertEqual(fedavg["attacker"], "decaf")
        self.assertEqual(fedlc["attacker"], "row")

    def test_pair_ridge_tracks_a_row_that_determines_the_count(self):
        rng = np.random.RandomState(0)
        row = rng.normal(size=(40, 3))
        bias = rng.normal(size=(40, 3))
        counts = 4.0 * row + 1.5
        _scaler, model = _fit_pair(row, bias, counts)
        predicted = _apply_pair(_scaler, model, row, bias)
        self.assertEqual(predicted.shape, counts.shape)
        self.assertGreater(counts_r2(counts, predicted), 0.95)


if __name__ == "__main__":
    unittest.main()
