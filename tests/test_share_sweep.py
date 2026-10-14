"""The share-bin bootstrap has to keep each client's features with that client's mix."""

import unittest

import numpy as np

from fedseismic.privacy.share_sweep import _mean_score


class ShareSweepBootstrapTests(unittest.TestCase):
    def test_one_draw_uses_the_same_clients_for_features_and_labels(self):
        seeds = [{
            "scaled": np.zeros((6, 1)),
            "mixes": np.zeros((6, 1)),
            "real": np.arange(6, dtype=np.float64).reshape(-1, 1),
            "truths": np.arange(6, dtype=np.float64).reshape(-1, 1),
        }]
        seen = []

        def scorer(_scaled, _mixes, real, truths, _lo, _hi):
            seen.append((np.asarray(real).ravel().copy(), np.asarray(truths).ravel().copy()))
            return {"accuracy": 1.0, "auc": 1.0, "n_real_absent": 1, "n_real_bin": 1}

        rng = np.random.RandomState(0)

        def index_of(seed, _rng=rng):
            n = len(seed["truths"])
            return _rng.randint(0, n, size=n)

        _mean_score(seeds, scorer, 0.0, 0.03, index_of=index_of)
        real, truths = seen[0]
        self.assertGreater(len(np.unique(real)), 1)
        self.assertTrue(np.array_equal(real, truths))


if __name__ == "__main__":
    unittest.main()
