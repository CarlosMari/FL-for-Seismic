"""Regression tests for per-survey heads (fedpersurvey).

The critical invariant (bug fixed 2026-10-20): on download the model must take
the per-survey pooled HEAD from the aggregate and restore ONLY local BN. The
head must NOT come from ``personal_state`` (that makes it per-client and throws
away the survey pooling), and BN must NOT be left at the global placeholder
(that drops the FedBN local-normalisation benefit and freezes BN at its initial
value). ``FedPerSurveyBNClientTrainer._local_keys`` must therefore be exactly the
BN keys, distinct from ``personalization_keys`` (the head) which the inherited
``FedPerClientTrainer.download`` would have overlaid.
"""

import unittest

import torch

from fedseismic.federated.client import _is_bn_key
from fedseismic.federated.fedper import personalization_keys
from fedseismic.federated.survey_heads import (
    FedPerSurveyBNClientTrainer,
    aggregate_survey_heads,
    survey_rows_from_info,
)


def _tiny_state(c=4, d=3, marker=0.0):
    return {
        "enc.weight": torch.full((d,), float(marker)),
        "outc.conv.weight": torch.full((c, d), float(marker)),
        "outc.conv.bias": torch.full((c,), float(marker)),
        "inc.double_conv.1.weight": torch.full((1,), float(marker)),  # real BN key
    }


class TestSurveyRows(unittest.TestCase):
    def test_rows_follow_class_fracs(self):
        self.assertEqual(
            survey_rows_from_info({"class_fracs": [1, 1, 0, 0]}, 4), [0, 1]
        )
        self.assertEqual(
            survey_rows_from_info({"class_fracs": [0, 0, 1, 1]}, 4), [2, 3]
        )


class TestAggregateSurveyHeads(unittest.TestCase):
    def test_rows_pool_within_survey_body_pools_all(self):
        s0 = _tiny_state(marker=1.0)
        s1 = _tiny_state(marker=2.0)
        s0["outc.conv.bias"][0] = 10.0  # site0 trains row0
        s1["outc.conv.bias"][1] = 20.0  # site1 trains row1
        infos = [{"class_fracs": [1, 0, 0, 0]}, {"class_fracs": [0, 1, 0, 0]}]
        out = aggregate_survey_heads([s0, s1], infos, 4)
        self.assertAlmostEqual(float(out["enc.weight"][0]), 1.5)  # body: all sites
        self.assertAlmostEqual(float(out["outc.conv.bias"][0]), 10.0)  # own site only
        self.assertAlmostEqual(float(out["outc.conv.bias"][1]), 20.0)
        self.assertAlmostEqual(float(out["inc.double_conv.1.weight"][0]), 1.0)  # BN untouched

    def test_weight_rows_pool_within_survey(self):
        s0 = _tiny_state(marker=0.0)
        s1 = _tiny_state(marker=0.0)
        s0["outc.conv.weight"][0] = torch.tensor([1.0, 1.0, 1.0])
        s1["outc.conv.weight"][1] = torch.tensor([2.0, 2.0, 2.0])
        infos = [{"class_fracs": [1, 0, 0, 0]}, {"class_fracs": [0, 1, 0, 0]}]
        out = aggregate_survey_heads([s0, s1], infos, 4)
        self.assertAlmostEqual(float(out["outc.conv.weight"][0][0]), 1.0)
        self.assertAlmostEqual(float(out["outc.conv.weight"][1][0]), 2.0)


class TestLocalKeysAreBNOnly(unittest.TestCase):
    def test_head_is_not_local_for_fedpersurvey(self):
        t = FedPerSurveyBNClientTrainer.__new__(FedPerSurveyBNClientTrainer)
        sd = _tiny_state()
        local = t._local_keys(sd)
        # local keys are exactly the BN keys (head is shared within survey)
        self.assertTrue(all(_is_bn_key(k) for k in local))
        self.assertFalse(any(k.startswith("outc.") for k in local))
        # personalization_keys (what FedPerClientTrainer.download overlays) is
        # the head — the two must be disjoint or download would restore the head
        # from personal_state and discard the per-survey pooling.
        head = set(personalization_keys(sd, None))
        self.assertTrue(any(k.startswith("outc.") for k in head))
        self.assertFalse(head & local)


if __name__ == "__main__":
    unittest.main()
