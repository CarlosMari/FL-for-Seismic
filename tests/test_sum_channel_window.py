"""Windowed victim channels: rounds 1-10 mean differs from all-round mean."""

import unittest

import numpy as np

from fedseismic.privacy.sum_channel import participation_channels


class WindowedChannelTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        (self.run / "head").mkdir(parents=True)
        rng = np.random.RandomState(0)
        for round_index in range(1, 41):
            scale = 1.0 if round_index <= 10 else 0.01
            weight = (rng.randn(4, 8) * scale).astype(np.float32)
            bias = (rng.randn(4) * scale).astype(np.float32)
            np.savez_compressed(
                self.run / "head" / f"round_{round_index:03d}_client_000.npz",
                weight_delta=weight, bias_delta=bias,
            )

    def tearDown(self):
        self.tmp.cleanup()

    def test_early_window_larger_than_full_mean(self):
        _, early_bias, _ = participation_channels(self.run, 1, 10)
        _, full_bias, _ = participation_channels(self.run, 1, 40)
        self.assertGreater(
            float(np.abs(early_bias[0]).mean()), float(np.abs(full_bias[0]).mean()))

    def test_empty_window_raises(self):
        with self.assertRaises(FileNotFoundError):
            participation_channels(self.run, 41, 50)


if __name__ == "__main__":
    unittest.main()
