"""Centering and round lookup for the body probe."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fedseismic.privacy.body_channel import center_rows, last_rounds


class BodyChannelTests(unittest.TestCase):
    def test_centering_removes_the_client_mean(self):
        scores = [np.array([1.0, 0.0]), np.array([3.0, 2.0])]
        centered = center_rows(scores)
        self.assertTrue(np.allclose(centered.mean(axis=0), 0.0))
        self.assertAlmostEqual(float(centered[1, 0] - centered[0, 0]), 2.0)

    def test_last_round_matches_the_received_file_index(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "round_scores.jsonl"
            path.write_text(
                "\n".join([
                    json.dumps({"client": 3, "round": 2}),
                    json.dumps({"client": 3, "round": 9}),
                    json.dumps({"client": 1, "round": 4}),
                ]),
                encoding="utf-8",
            )
            self.assertEqual(last_rounds(folder), {3: 9, 1: 4})


if __name__ == "__main__":
    unittest.main()
