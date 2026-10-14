"""Presence metrics resample clients, not individual class rows."""

import unittest

import pandas as pd

from fedseismic.privacy.round_scores import cluster_bootstrap_metrics, count_bin


class PresenceBootstrapTests(unittest.TestCase):
    def test_count_bins(self):
        self.assertEqual(count_bin(0), "0")
        self.assertEqual(count_bin(2), "1-2")
        self.assertEqual(count_bin(10), "3-10")
        self.assertEqual(count_bin(11), ">10")

    def test_separated_residuals_stay_high(self):
        rows = []
        for client in range(8):
            for class_index in range(2):
                present = int(class_index == 0)
                rows.append({
                    "seed": 0,
                    "client": client,
                    "present": present,
                    "residual": 1.0 if present else 0.0,
                })
        result = cluster_bootstrap_metrics(pd.DataFrame(rows), draws=50)
        self.assertEqual(result["n_clusters"], 8)
        self.assertGreater(result["auc"][0], 0.9)


if __name__ == "__main__":
    unittest.main()
