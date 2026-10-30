"""FedHCA2-style aggregation: update math, what stays local, and a CPU smoke run."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from fedseismic.config import RunConfig
from fedseismic.experiment import run
from fedseismic.federated.client import _is_bn_key
from fedseismic.federated.fedhca2 import (
    FedHCA2Aggregator, FedHCA2ClientTrainer, cross_attention_deltas, decoder_keys,
    encoder_keys, get_ca_delta,
)
from fedseismic.federated.server import Server
from fedseismic.models import UNet
from fedseismic.privacy.multisurvey_utility import rescore_run

_spec = importlib.util.spec_from_file_location(
    "compare_k3", Path(__file__).parents[1] / "scripts" / "compare_multisurvey3_k3.py")
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)


class TestConflictAverse(unittest.TestCase):
    def test_identical_deltas_have_closed_form(self):
        g = torch.tensor([3.0, 4.0])
        out = get_ca_delta([g, g], 0.4)
        self.assertTrue(torch.allclose(out, g * 1.4 / 1.16, atol=1e-3))

    def test_zero_deltas_give_zero(self):
        out = get_ca_delta([torch.zeros(3), torch.zeros(3)], 0.4)
        self.assertTrue(torch.equal(out, torch.zeros(3)))

    def test_update_improves_worst_client_over_mean(self):
        # g1 is small and conflicts with g0, so the plain mean moves against g1.
        g0, g1 = torch.tensor([10.0, 0.0]), torch.tensor([-1.0, 1.0])
        out = get_ca_delta([g0, g1], 0.9)
        mean = (g0 + g1) / 2
        self.assertLess(torch.dot(mean, g1), 0.0)
        self.assertGreater(torch.dot(out, g1) / out.norm(), torch.dot(mean, g1) / mean.norm())


class TestCrossAttention(unittest.TestCase):
    def test_matches_upstream_per_client_loop(self):
        deltas = torch.randn(3, 8, generator=torch.Generator().manual_seed(0))
        out = cross_attention_deltas(deltas)
        for i in range(3):
            q = deltas[i].reshape(1, -1)
            attn = torch.softmax((q @ deltas.t()) * 8 ** -0.5, dim=-1)
            self.assertTrue(torch.allclose(out[i], (attn @ deltas)[0]))


class TestFedHCA2HeadPath(unittest.TestCase):
    def test_fedhca2_server_does_not_use_survey_head_aggregator(self):
        server = Server.__new__(Server)
        server.config = SimpleNamespace(algorithm="fedhca2", agg_strategy="equal")
        w = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1, 1)
        with patch("fedseismic.federated.survey_heads.aggregate_survey_heads",
                   side_effect=AssertionError("wrong aggregation path")):
            out = server._aggregate([{"outc.conv.weight": w},
                                     {"outc.conv.weight": w + 2}], [0, 1])
        self.assertTrue(torch.equal(out["outc.conv.weight"], w + 1))

    def test_download_restores_nonconstant_personal_head(self):
        trainer = FedHCA2ClientTrainer.__new__(FedHCA2ClientTrainer)
        trainer.model = torch.nn.Conv2d(3, 2, kernel_size=1)
        trainer.device = "cpu"
        personal = trainer.model.state_dict()
        personal["weight"] = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1, 1)
        trainer.personal_state = personal
        placeholder = {key: torch.zeros_like(value) for key, value in personal.items()}
        trainer.download(placeholder)
        self.assertTrue(torch.equal(trainer.model.weight.detach(), personal["weight"]))
        self.assertGreater(float(trainer.model.weight[0].detach().std()), 0.0)


def _unet_state():
    torch.manual_seed(0)
    return UNet(1, 19, bilinear=False, norm="batch", groups=8).state_dict()


def _perturbed(state, keys, seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        key: (value + 0.01 * torch.randn(value.shape, generator=generator)) if key in keys else value
        for key, value in state.items()
    }


class TestAggregator(unittest.TestCase):
    def setUp(self):
        self.state = _unet_state()
        self.shared = encoder_keys(self.state) + decoder_keys(self.state)

    def test_head_and_bn_are_not_shared(self):
        self.assertFalse(any(key.startswith(("outc.", "reconstruct.")) for key in self.shared))
        self.assertFalse(any(_is_bn_key(key) for key in self.shared))
        self.assertIn("inc.double_conv.0.weight", self.shared)
        self.assertIn("up1.up.weight", self.shared)

    def test_zero_alpha_beta_leaves_each_client_at_its_own_weights(self):
        agg = FedHCA2Aggregator(self.state, 3, alpha_init=0.0, beta_init=0.0)
        trained = [_perturbed(self.state, self.shared, seed) for seed in range(3)]
        for client, new in enumerate(agg.step(trained)):
            for key in self.shared:
                self.assertTrue(torch.allclose(new[key], trained[client][key].float(), atol=1e-5))

    def test_nonzero_weights_share_information_but_stay_personal(self):
        agg = FedHCA2Aggregator(self.state, 3)
        trained = [_perturbed(self.state, self.shared, seed) for seed in range(3)]
        new = agg.step(trained)
        key = "inc.double_conv.0.weight"
        self.assertFalse(torch.allclose(new[0][key], trained[0][key]))
        self.assertFalse(torch.allclose(new[0][key], new[1][key]))

    def test_hyper_gradient_matches_autograd(self):
        keys = {"enc": ["inc.double_conv.0.weight"], "dec": ["up1.up.weight"]}
        agg = FedHCA2Aggregator(self.state, 2, lr=1.0, momentum=0.0, weight_decay=0.0)
        first = [_perturbed(self.state, self.shared, seed) for seed in range(2)]
        agg.step(first)
        update = agg._encoder_update.clone()
        attended = [dict(agg._decoder_updates[i]) for i in range(2)]
        last = agg._last
        second = [_perturbed(self.state, self.shared, seed + 5) for seed in range(2)]
        alpha0 = agg.alpha.detach().clone()
        layer = keys["dec"][0].rsplit(".", 1)[0]
        beta0 = agg.beta[layer].detach().clone()

        # Upstream's rule on the same numbers: backward(output, last - trained).
        alpha = alpha0.clone().requires_grad_()
        flat_last = torch.cat([last[0][k].reshape(-1) for k in agg.encoder_keys])
        flat_new = torch.cat([second[0][k].reshape(-1).float() for k in agg.encoder_keys])
        output = flat_last + torch.clamp(alpha, 0, 1)[0] * update
        torch.autograd.backward(output, flat_last - flat_new)
        beta = beta0.clone().requires_grad_()
        output = last[1][keys["dec"][0]] + torch.clamp(beta, 0, 1)[1] * attended[1][keys["dec"][0]]
        diff = last[1][keys["dec"][0]] - second[1][keys["dec"][0]].float()
        torch.autograd.backward(output, diff)
        beta_grad = beta.grad[1].item()
        # The layer has weight and bias; add the bias term like the aggregator does.
        bias_key = f"{layer}.bias"
        beta_grad += torch.dot(
            attended[1][bias_key].reshape(-1),
            (last[1][bias_key] - second[1][bias_key].float()).reshape(-1),
        ).item()

        agg.step(second)
        self.assertAlmostEqual((alpha0[0] - agg.alpha[0]).item(), alpha.grad[0].item(), places=2)
        self.assertAlmostEqual((beta0[1] - agg.beta[layer][1]).item(), beta_grad, places=2)


def _cube(path_x, path_y, low, high, seed, ignore=None):
    rng = np.random.RandomState(seed)
    # (inline, crossline, depth): sections must survive the UNet's four poolings.
    labels = rng.randint(low, high, (32, 8, 32)).astype(np.int64)
    if ignore is not None:
        labels[:, :, :2] = ignore
    # Every class is guaranteed in the train part of the crossline axis.
    labels[:, 0, 2: 2 + high - low] = np.arange(low, high)
    np.save(path_x, rng.randn(32, 8, 32).astype(np.float64))
    np.save(path_y, labels)


class TestSmokeRun(unittest.TestCase):
    def test_three_surveys_train_and_rescore_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _cube(tmp / "ax.npy", tmp / "ay.npy", 0, 6, 0)
            _cube(tmp / "bx.npy", tmp / "by.npy", 1, 7, 1)
            _cube(tmp / "cx.npy", tmp / "cy.npy", 1, 8, 2, ignore=0)
            out = tmp / "run"
            cfg = RunConfig.from_mapping({
                "multisurvey": True, "num_clients": 3, "num_rounds": 3, "local_epochs": 1,
                "batch_size": 2, "num_classes": 19, "algorithm": "fedhca2", "device": "cpu",
                "train_seismic": str(tmp / "ax.npy"), "train_labels": str(tmp / "ay.npy"),
                "survey_b_seismic": str(tmp / "bx.npy"), "survey_b_labels": str(tmp / "by.npy"),
                "survey_c_seismic": str(tmp / "cx.npy"), "survey_c_labels": str(tmp / "cy.npy"),
                "survey_c_ignore": 0, "num_workers": 0, "output_dir": str(out),
                "privacy_log_last_only": True, "privacy_save_local": False,
            })
            run(cfg, seeds=[0])
            seed_dir = out / "seed_0"
            self.assertTrue((seed_dir / "results.json").is_file())
            weights = [json.loads(line) for line in (seed_dir / "hyper_weights.jsonl").read_text().splitlines()]
            self.assertEqual(len(weights), 3)
            self.assertEqual(len(weights[0]["alpha"]), 3)
            self.assertNotEqual(weights[0]["alpha"], weights[-1]["alpha"])

            states = [torch.load(seed_dir / "personal" / f"client_{c:03d}.pt") for c in range(3)]
            bias = "outc.conv.bias"
            self.assertEqual(int((states[0][bias] > -1e8).sum()), 6)
            self.assertTrue(bool((states[1][bias][6:12] > -1e8).all()))
            self.assertEqual(int((states[2][bias] > -1e8).sum()), 7)
            body = "inc.double_conv.0.weight"
            self.assertFalse(torch.equal(states[0][body], states[1][body]))
            bn = "inc.double_conv.1.running_mean"
            self.assertFalse(torch.equal(states[0][bn], states[1][bn]))

            scored = rescore_run(seed_dir, device="cpu")
            for code in "ABC":
                self.assertTrue(np.isfinite(scored[f"{code}_local_miou"]))
            self.assertTrue(np.isnan(scored["A_global_on_local"]))
            rows = compare.rows_from_score("fedhca2", 0, seed_dir, scored)
            self.assertEqual([row["survey"] for row in rows], ["F3", "Parihaka", "Penobscot"])

    def test_more_than_one_client_per_survey_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _cube(tmp / "ax.npy", tmp / "ay.npy", 0, 6, 0)
            _cube(tmp / "bx.npy", tmp / "by.npy", 1, 7, 1)
            cfg = RunConfig.from_mapping({
                "multisurvey": True, "num_clients": 4, "num_classes": 12, "algorithm": "fedhca2",
                "train_seismic": str(tmp / "ax.npy"), "train_labels": str(tmp / "ay.npy"),
                "survey_b_seismic": str(tmp / "bx.npy"), "survey_b_labels": str(tmp / "by.npy"),
                "num_workers": 0,
            })
            with self.assertRaisesRegex(ValueError, "one client per survey"):
                run(cfg, seeds=[0])


class TestCompareTable(unittest.TestCase):
    def test_all_class_uses_survey_block_not_union(self):
        scored = {
            "A_local_miou": 0.5, "B_local_miou": float("nan"),
            "C_local_miou": float("nan"), "A_local_all_miou": 3.0 / 19,
            "_per_site_class_rows": [
                {"survey": "A", "class": k, "iou": 0.5} for k in range(6)
            ],
        }
        row = compare.rows_from_score("solo", 0, "run", scored)[0]
        self.assertAlmostEqual(row["miou_all_class"], 0.5)
        self.assertAlmostEqual(row["miou_union_all_class"], 3.0 / 19)

    def test_delta_is_paired_by_survey_and_seed(self):
        frame = pd.DataFrame([
            {"arm": "solo", "seed": 0, "survey": "F3", "miou_all_class": 0.5, "miou_present_class": 0.5},
            {"arm": "solo", "seed": 1, "survey": "F3", "miou_all_class": 0.7, "miou_present_class": 0.7},
            {"arm": "fedhca2", "seed": 0, "survey": "F3", "miou_all_class": 0.6, "miou_present_class": 0.6},
            {"arm": "fedhca2", "seed": 1, "survey": "F3", "miou_all_class": 0.6, "miou_present_class": 0.6},
        ])
        out = compare.add_solo_delta(frame)
        fl = out[out["arm"] == "fedhca2"].sort_values("seed")["delta_vs_solo"].tolist()
        self.assertAlmostEqual(fl[0], 0.1)
        self.assertAlmostEqual(fl[1], -0.1)
        self.assertTrue(out[out["arm"] == "solo"]["delta_vs_solo"].isna().all())
        summary = compare.summarize(out)
        row = summary[summary["arm"] == "fedhca2"].iloc[0]
        self.assertEqual(row["n_seeds"], 2)
        self.assertAlmostEqual(row["delta_vs_solo"], 0.0)


if __name__ == "__main__":
    unittest.main()
