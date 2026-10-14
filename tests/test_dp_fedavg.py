"""Client-level DP-FedAvg clips the delta and noises the sum."""

import unittest

import numpy as np
import torch

from fedseismic.federated.aggregation import dp_fedavg_aggregate


class DPFedAvgTests(unittest.TestCase):
    def test_clip_bounds_the_update_when_noise_is_off(self):
        global_state = {"weight": torch.zeros(4)}
        uploaded = [{"weight": torch.ones(4) * 10}]
        updated = dp_fedavg_aggregate(
            global_state, uploaded, clip_norm=2.0, noise_multiplier=0.0,
            rng=np.random.RandomState(0),
        )
        delta = updated["weight"] - global_state["weight"]
        self.assertLessEqual(float(delta.norm()), 2.0 + 1e-5)

    def test_noise_changes_the_aggregate(self):
        global_state = {"weight": torch.zeros(8)}
        uploaded = [{"weight": torch.ones(8)}]
        quiet = dp_fedavg_aggregate(
            global_state, uploaded, clip_norm=5.0, noise_multiplier=0.0,
            rng=np.random.RandomState(0),
        )
        noisy = dp_fedavg_aggregate(
            global_state, uploaded, clip_norm=5.0, noise_multiplier=1.0,
            rng=np.random.RandomState(0),
        )
        self.assertFalse(torch.allclose(quiet["weight"], noisy["weight"]))


if __name__ == "__main__":
    unittest.main()
