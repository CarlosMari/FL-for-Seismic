"""Decaf's hard rule and the shadow calibration of the positive-sum score."""

import unittest

import numpy as np

from fedseismic.privacy.decaf import positive_sum, row_has_positive, shadow_proportion_r2


class HardRuleTests(unittest.TestCase):
    def test_only_a_positive_entry_marks_the_row_present(self):
        delta = np.array([[-1.0, -0.2], [0.1, -0.3]])
        present = row_has_positive(delta)
        self.assertFalse(bool(present[0]))
        self.assertTrue(bool(present[1]))
        sums = positive_sum(delta)
        self.assertAlmostEqual(float(sums[0]), 0.0)
        self.assertAlmostEqual(float(sums[1]), 0.1)


class ProportionTests(unittest.TestCase):
    def test_shadow_line_recovers_a_planted_share(self):
        shares = np.array([0.1, 0.2, 0.4, 0.7, 0.3, 0.5])
        real = [{
            "pi": np.array([0.25, 0.75]),
            "score": np.array([0.25, 0.75]),
            "hard_present": np.array([True, True]),
        }]
        scored = shadow_proportion_r2(real, shares, shares)
        self.assertGreater(scored["r2"], 0.99)

    def test_absent_class_is_removed_before_the_simplex(self):
        shares = np.array([0.1, 0.2, 0.4, 0.7, 0.3, 0.5])
        real = [{
            "pi": np.array([0.0, 1.0]),
            "score": np.array([0.4, 0.6]),
            "hard_present": np.array([False, True]),
        }]
        scored = shadow_proportion_r2(real, shares, shares)
        self.assertGreater(scored["r2"], 0.99)


if __name__ == "__main__":
    unittest.main()
