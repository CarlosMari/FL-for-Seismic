"""Logit-offset and classifier-row formulas."""

import unittest

import numpy as np

from fedseismic.privacy.attack_suite import classifier_row_norms, offset_prior


class AttackSuiteTests(unittest.TestCase):
    def test_offset_recovers_log_prior(self):
        pi = np.array([0.7, 0.2, 0.1])
        hat = offset_prior(np.log(pi), np.zeros(3))
        np.testing.assert_allclose(hat, pi, atol=1e-6)

    def test_matching_rows_have_zero_norm(self):
        weight = np.arange(12, dtype=np.float64).reshape(3, 4)
        norms = classifier_row_norms(weight, weight)
        np.testing.assert_allclose(norms, np.zeros(3))


if __name__ == "__main__":
    unittest.main()
