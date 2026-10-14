"""Sum-channel helpers: cohort mean, peer residual, windowed channels."""

import unittest

import numpy as np

from fedseismic.privacy.sum_channel import cohort_sum


class SumChannelTests(unittest.TestCase):
    def test_cohort_sum_is_the_mean(self):
        channels = {0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])}
        n, summed = cohort_sum(channels)
        self.assertEqual(n, 2)
        self.assertTrue(np.allclose(summed, [2.0, 3.0]))

    def test_peer_residual_recovers_other_member_at_n2(self):
        channels = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}
        n, summed = cohort_sum(channels)
        real = np.stack([channels[0], channels[1]])
        residual = (summed * n - real[0]) / (n - 1)
        self.assertTrue(np.allclose(residual, channels[1]))


if __name__ == "__main__":
    unittest.main()
