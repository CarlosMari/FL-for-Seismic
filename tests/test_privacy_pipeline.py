"""Unit tests for FedKPer diversity modes and the privacy–personalization pipeline."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fedseismic.config import RunConfig
from fedseismic.experiment import FederatedData
from fedseismic.federated.aggregation import FedKPerAgg, get_agg_weights, normalized_entropy
from fedseismic.federated.client import class_frequency
from fedseismic.federated.server import Server
from fedseismic.privacy.attacks import evaluate_run
from fedseismic.privacy.estimators import last_layer_prior, softmax_prior
from fedseismic.privacy.log import PrivacyLogger, load_labels, load_views
from fedseismic.privacy.metrics import leakage_from_tv, rare_presence_f1, total_variation
from fedseismic.privacy.plot import pareto_mask
from fedseismic.privacy.sweep import expand_jobs, load_sweep_spec


class TripleDataset(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets
        self.indices = list(range(len(targets)))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], self.targets[index], index


class TinyNet(nn.Module):
    def __init__(self, num_classes=3, dim=4):
        super().__init__()
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.classifier(x.view(x.size(0), -1))


def _client_info(loaders, num_classes):
    info = []
    for loader in loaders:
        fracs = class_frequency(loader, num_classes)
        present = {index for index, value in enumerate(fracs) if value > 0}
        info.append({
            "num_classes": len(present),
            "has_classes": present,
            "rare_fraction": float(sum(fracs[index] for index in (1, 2) if index < num_classes)),
            "class_fracs": fracs,
        })
    return info


def _synthetic_run(diversity, tmp, extra=None):
    torch.manual_seed(0)
    np.random.seed(0)
    images0 = torch.randn(8, 4)
    images1 = torch.randn(8, 4)
    targets0 = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
    targets1 = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2])
    loaders = [
        DataLoader(TripleDataset(images0, targets0), batch_size=4, shuffle=False),
        DataLoader(TripleDataset(images1, targets1), batch_size=4, shuffle=False),
    ]
    tests = [
        DataLoader(TripleDataset(images0[:4], targets0[:4]), batch_size=4, shuffle=False),
        DataLoader(TripleDataset(images1[:4], targets1[:4]), batch_size=4, shuffle=False),
    ]
    probe = DataLoader(
        TripleDataset(torch.randn(12, 4), torch.zeros(12, dtype=torch.long)),
        batch_size=4, shuffle=False,
    )
    kwargs = dict(
        dataset="bloodmnist", num_clients=2, num_rounds=1, local_epochs=1,
        batch_size=4, num_classes=3, sample_ratio=1.0, split="iid",
        algorithm="fedkper", agg_strategy="fedkper", fedkper_diversity=diversity,
        device="cpu", lr=0.05, optimizer="sgd", output_dir=str(tmp),
        privacy_delta_dim=16, privacy_probe_batches=2, privacy_save_local=True,
        rare_classes=(1, 2),
    )
    if extra:
        kwargs.update(extra)
    cfg = RunConfig(**kwargs)
    logger = PrivacyLogger(
        tmp, projection_dim=cfg.privacy_delta_dim, projection_seed=0, save_local=True,
    )
    info = _client_info(loaders, 3)
    logger.write_ground_truth([row["class_fracs"] for row in info], rare_classes=(1, 2))
    factory = lambda: TinyNet(3, 4)
    server = Server(
        model=factory(), client_loaders=loaders, criterion=nn.CrossEntropyLoss(),
        config=cfg, client_info=info, model_factory=factory, client_test_loaders=tests,
        test_loaders={"test1": probe}, probe_loader=probe, privacy_logger=logger,
        rng=np.random.RandomState(0),
    )
    history = server.run()
    logger.write_meta({
        "seed": 0,
        "config": cfg.to_dict(),
        "utility": {
            "local_mean": history[-1].miou_local_mean,
            "local_worst": history[-1].miou_local_worst,
            "global_on_local": history[-1].miou_global_on_local,
            "global_score": history[-1].miou_final,
        },
    })
    data = FederatedData(
        train_labels=np.concatenate([targets0.numpy(), targets1.numpy()]),
        train_parts=[list(range(8)), list(range(8, 16))],
        loaders=loaders, client_tests=tests, tests={"test1": probe},
        model_factory=factory, client_info=info,
    )
    return cfg, server, data, info


class MetricsTests(unittest.TestCase):
    def test_tv_and_leakage(self):
        pi = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(total_variation(pi, pi), 0.0)
        self.assertAlmostEqual(total_variation(pi, np.array([0.0, 1.0, 0.0])), 1.0)
        self.assertAlmostEqual(leakage_from_tv(0.25), 0.75)

    def test_rare_presence_f1(self):
        true = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0])
        hat = np.array([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(rare_presence_f1(true, hat, (4, 5)), 1.0)
        hat_wrong = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
        self.assertEqual(rare_presence_f1(true, hat_wrong, (4, 5)), 0.0)

    def test_normalized_entropy_uniform(self):
        self.assertAlmostEqual(normalized_entropy([0.25, 0.25, 0.25, 0.25]), 1.0, places=5)


class EstimatorTests(unittest.TestCase):
    def test_last_layer_prefers_large_rows(self):
        weight = torch.zeros(3, 4)
        weight[2] = 5.0
        prior = last_layer_prior({"classifier.weight": weight}, num_classes=3)
        self.assertEqual(int(np.argmax(prior)), 2)

    def test_softmax_prior_biased_model(self):
        model = TinyNet(3, 4)
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.copy_(torch.tensor([5.0, 0.0, 0.0]))
        loader = DataLoader(
            TripleDataset(torch.zeros(6, 4), torch.zeros(6, dtype=torch.long)),
            batch_size=3,
        )
        prior = softmax_prior(model, loader, 3, "cpu", max_batches=2)
        self.assertGreater(prior[0], 0.8)


class AggregationTests(unittest.TestCase):
    def test_oracle_matches_explicit_histograms(self):
        client_info = [
            {"class_fracs": np.array([1.0, 0.0, 0.0])},
            {"class_fracs": np.array([0.0, 0.5, 0.5])},
        ]
        accs = [0.4, 0.8]
        oracle = FedKPerAgg().weights([0, 1], client_info, client_train_accs=accs)
        explicit = FedKPerAgg().weights(
            [0, 1], client_info, client_train_accs=accs,
            client_histograms=[client_info[0]["class_fracs"], client_info[1]["class_fracs"]],
        )
        np.testing.assert_allclose(oracle, explicit)
        via_helper = get_agg_weights(
            "fedkper", [0, 1], client_info, client_train_accs=accs,
        )
        np.testing.assert_allclose(oracle, via_helper)


class LoggerTests(unittest.TestCase):
    def test_views_do_not_include_true_pi(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = PrivacyLogger(tmp, projection_dim=8, projection_seed=1)
            true = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
            logger.write_ground_truth(true, rare_classes=(1,))
            global_state = {"classifier.weight": torch.zeros(2, 2), "classifier.bias": torch.zeros(2)}
            uploaded = {"classifier.weight": torch.ones(2, 2), "classifier.bias": torch.ones(2)}
            logger.log_upload(1, 0, uploaded, global_state, scalars={"train_acc": 0.5})
            views = load_views(tmp)
            self.assertEqual(len(views), 1)
            self.assertNotIn("pi", views[0])
            labels, rare = load_labels(tmp)
            self.assertEqual(rare, (1,))
            np.testing.assert_allclose(labels[0], true[0])


class PipelineTests(unittest.TestCase):
    def test_upload_matches_oracle_weights_on_synthetic(self):
        with tempfile.TemporaryDirectory() as oracle_dir, tempfile.TemporaryDirectory() as upload_dir:
            _, oracle_server, _, info = _synthetic_run("oracle", oracle_dir)
            _, upload_server, _, _ = _synthetic_run("upload", upload_dir)
            accs = [0.5, 0.5]
            hists = [row["class_fracs"] for row in info]
            oracle_w = FedKPerAgg().weights([0, 1], info, client_train_accs=accs)
            upload_w = FedKPerAgg().weights(
                [0, 1], info, client_train_accs=accs, client_histograms=hists,
            )
            np.testing.assert_allclose(oracle_w, upload_w)
            self.assertEqual(oracle_server.config.fedkper_diversity, "oracle")
            self.assertEqual(upload_server.config.fedkper_diversity, "upload")

    def test_infer_and_attack_on_synthetic_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, _, data, _ = _synthetic_run("infer", tmp)
            result = evaluate_run(tmp, data=data, device="cpu", max_probe_batches=2, max_mia_examples=8)
            self.assertIn("softmax", result)
            self.assertGreaterEqual(result["softmax"]["tv"], 0.0)
            self.assertLessEqual(result["softmax"]["tv"], 1.0)
            self.assertTrue(np.isfinite(result["last_layer"]["leakage"]))
            self.assertEqual(cfg.fedkper_diversity, "infer")
            views = load_views(tmp)
            self.assertTrue(views)
            sidecar = json.loads(next(Path(tmp).joinpath("view").glob("*.json")).read_text())
            self.assertIn("protocol_histogram", sidecar)

    def test_sweep_expansion_and_seismic_spec(self):
        spec = load_sweep_spec("configs/sweeps/privacy_boundary.json")
        jobs = expand_jobs(spec)
        algorithms = {job["algorithm"] for job in jobs}
        self.assertEqual(algorithms, {"fedavg", "fedprox", "fedbn", "fedkper"})
        kper = [job for job in jobs if job["algorithm"] == "fedkper"]
        self.assertTrue(any(job["fedkper_diversity"] == "infer" for job in kper))
        self.assertTrue(any(job["fedkper_diversity"] == "upload" for job in kper))
        self.assertTrue(any(job["fedkper_diversity"] == "oracle" for job in kper))
        seismic = expand_jobs(load_sweep_spec("configs/sweeps/privacy_boundary_seismic.json"))
        self.assertTrue(any(job["algorithm"] == "fedkper" for job in seismic))

    def test_pareto_mask(self):
        leakage = np.array([0.9, 0.2, 0.2])
        gap = np.array([0.1, 0.4, 0.1])
        mask = pareto_mask(leakage, gap)
        self.assertTrue(mask[1])
        self.assertFalse(mask[0])


class ConfigTests(unittest.TestCase):
    def test_rejects_unknown_diversity(self):
        with self.assertRaises(ValueError):
            RunConfig(fedkper_diversity="secret")

    def test_default_is_infer(self):
        self.assertEqual(RunConfig().fedkper_diversity, "infer")


if __name__ == "__main__":
    unittest.main()
