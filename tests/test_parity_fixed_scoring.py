"""Verify the corrected parity report preserves survey blocks and support."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "score_parity_fixed", Path(__file__).parents[1] / "scripts" / "score_parity_fixed.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


class TestFixedParityReport(unittest.TestCase):
    def test_scores_separate_union_rows_and_training_support(self):
        with tempfile.TemporaryDirectory() as directory:
            old_cwd = Path.cwd()
            try:
                os.chdir(directory)
                for arm in score.ARMS:
                    for seed in (0, 1):
                        run = Path("results/runs") / arm / "cell" / f"seed_{seed}"
                        (run / "labels").mkdir(parents=True)
                        (run / "results.json").write_text("{}")
                        a = [1.0] * 5 + [0.0] + [0.0] * 6
                        b = [0.0] * 6 + [1.0] * 6
                        clients = ([{"client": 0, "pi": a}, {"client": 1, "pi": b}]
                                   if arm == "parity_fl_fixed" else
                                   [{"client": 0, "pi": b if arm == "parity_pari_solo_fixed" else a}])
                        (run / "labels/pi.json").write_text(json.dumps({"clients": clients}))

                def fake_score(run, device):
                    arm = Path(run).parts[2]
                    codes = ("A", "B") if arm == "parity_fl_fixed" else (
                        ("B",) if arm == "parity_pari_solo_fixed" else ("A",))
                    out = {"A_local_miou": float("nan"), "B_local_miou": float("nan"),
                           "_per_site_class_rows": []}
                    for code in codes:
                        _, lo, hi = score.BLOCKS[code]
                        client = 1 if code == "B" and arm == "parity_fl_fixed" else 0
                        vals = [0.0 if k == 5 else 0.6 for k in range(lo, hi)]
                        out[f"{code}_local_miou"] = sum(vals) / len(vals)
                        out["_per_site_class_rows"].extend(
                            {"survey": code, "client": client, "class": k,
                             "iou": value, "present": True} for k, value in zip(range(lo, hi), vals))
                    return out

                with patch("fedseismic.privacy.multisurvey_utility.rescore_run", fake_score), \
                     patch("sys.argv", ["score_parity_fixed", "--device", "cpu"]):
                    score.main()
                runs = pd.read_csv("results/tables/parity_fixed_runs.csv")
                per = pd.read_csv("results/tables/parity_fixed_per_class.csv")
                row = runs[(runs.arm == "parity_f3_solo_fixed") & (runs.seed == 0)].iloc[0]
                self.assertAlmostEqual(row.present_class_miou, 0.5)
                self.assertAlmostEqual(row.train_seen_test_present_miou, 0.6)
                fifth = per[(per.arm == "parity_f3_solo_fixed") & (per.seed == 0) & (per.union_class == 5)].iloc[0]
                self.assertFalse(bool(fifth.train_present))
                self.assertTrue(bool(fifth.test_present))
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
