"""A round-boundary resume must reproduce uninterrupted FedHCA2 training."""

import json
import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fedseismic.config import RunConfig
from fedseismic.experiment import run
from fedseismic.federated.server import Server
from fedseismic.privacy.log import PrivacyLogger


class TinyDataset(Dataset):
    def __init__(self, first_class):
        self.indices = list(range(4))
        self.targets = np.empty((4, 4, 4), dtype=np.int64)
        for i in self.indices:
            self.targets[i].fill(first_class + i % 2)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image = torch.full((1, 4, 4), float(index) / 4)
        return image, torch.as_tensor(self.targets[index]), index


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)

    def forward(self, x):
        return self.conv(x)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = torch.nn.Conv2d(1, 3, 1)
        self.up1 = torch.nn.Conv2d(3, 3, 1)
        self.outc = TinyHead()

    def forward(self, x):
        return self.outc(torch.relu(self.up1(torch.relu(self.inc(x)))))


def make_server(directory, resume=None, device="cpu"):
    torch.manual_seed(123)
    np.random.seed(123)
    config = RunConfig.from_mapping({
        "dataset": "seismic", "multisurvey": True, "algorithm": "fedhca2",
        "num_clients": 2, "num_classes": 4, "num_rounds": 3, "local_epochs": 1,
        "batch_size": 2, "device": device, "save_round_checkpoint": True,
        "resume_round_checkpoint": str(resume) if resume else None,
        "privacy_log_last_only": True, "privacy_save_local": False,
    })
    loaders = [DataLoader(TinyDataset(first), batch_size=2, shuffle=True) for first in (0, 2)]
    infos = [{"class_fracs": np.array([0.5, 0.5, 0, 0])},
             {"class_fracs": np.array([0, 0, 0.5, 0.5])}]
    logger = PrivacyLogger(directory, log_last_only=True, save_local=False)
    server = Server(TinyModel(), loaders, torch.nn.CrossEntropyLoss(), config, device=device,
                    model_factory=TinyModel, client_info=infos,
                    rng=np.random.RandomState(123), privacy_logger=logger)
    server.run_seed = 123
    return server


class TestFedHCA2Checkpoint(unittest.TestCase):
    def test_multi_seed_resume_is_rejected(self):
        cfg = RunConfig.from_mapping({
            "dataset": "seismic", "multisurvey": True, "algorithm": "fedhca2",
            "num_clients": 2, "resume_round_checkpoint": "/tmp/does-not-matter.pt",
        })
        with self.assertRaisesRegex(ValueError, "exactly one seed"):
            run(cfg, seeds=[0, 1])

    def test_control_and_hyper_rate_variant_differ_only_in_hyper_rate(self):
        root = Path(__file__).parents[1]
        control = json.loads((root / "configs/seismic_multisurvey3_k3_recoverable.json").read_text())
        variant = json.loads((root / "configs/seismic_multisurvey3_k3_hlr1e4.json").read_text())
        changes = {key for key in set(control) | set(variant) if control.get(key) != variant.get(key)}
        self.assertEqual(changes, {"fedhca2_hyper_lr"})
        self.assertEqual(control["fedhca2_hyper_lr"], 0.01)
        self.assertEqual(variant["fedhca2_hyper_lr"], 0.0001)
        sweeps = [json.loads((root / "configs/sweeps" / name).read_text()) for name in (
            "multisurvey3_k3_fedhca2_restart_s0.json", "multisurvey3_k3_fedhca2_hlr1e4_s0.json")]
        self.assertEqual([s["seeds"] for s in sweeps], [[0], [0]])
        self.assertNotEqual(sweeps[0]["output_root"], sweeps[1]["output_root"])

    def test_interrupted_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = make_server(root / "full")
            full.run()
            full_torch_rng = torch.get_rng_state().clone()
            full_server_rng = full.rng.get_state()

            interrupted = make_server(root / "interrupted")
            original = interrupted.run_round

            def stop_after_one(index):
                if index == 1:
                    raise RuntimeError("simulated interruption")
                record = original(index)
                record.miou_local_mean = np.float64(0.25)
                record.per_class_iou = [np.float64(0.125)]
                return record

            with patch.object(interrupted, "run_round", side_effect=stop_after_one):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    interrupted.run()
            checkpoint = root / "interrupted" / "round_checkpoint.pt"
            self.assertTrue(checkpoint.is_file())

            resumed = make_server(root / "resumed", checkpoint)
            resumed.run()
            self.assertEqual([r.round for r in resumed.history], [1, 2, 3])
            self.assertIs(type(resumed.history[0].miou_local_mean), float)
            self.assertIs(type(resumed.history[0].per_class_iou[0]), float)
            for client in (0, 1):
                actual = resumed._trainer(client).personal_state
                expected = full._trainer(client).personal_state
                for key in actual:
                    self.assertTrue(torch.equal(actual[key], expected[key]), (client, key))
            self.assertTrue(torch.equal(resumed._fedhca2.alpha, full._fedhca2.alpha))
            for layer in full._fedhca2.beta:
                self.assertTrue(torch.equal(resumed._fedhca2.beta[layer], full._fedhca2.beta[layer]))
            self.assertTrue(torch.equal(torch.get_rng_state(), full_torch_rng))
            resumed_rng = resumed.rng.get_state()
            self.assertEqual(resumed_rng[0], full_server_rng[0])
            np.testing.assert_array_equal(resumed_rng[1], full_server_rng[1])
            self.assertEqual(resumed_rng[2:], full_server_rng[2:])
            full_opt = full._fedhca2.optimizer.state_dict()["state"]
            resumed_opt = resumed._fedhca2.optimizer.state_dict()["state"]
            self.assertEqual(full_opt.keys(), resumed_opt.keys())
            for key in full_opt:
                self.assertTrue(torch.equal(full_opt[key]["momentum_buffer"],
                                            resumed_opt[key]["momentum_buffer"]))
            self.assertEqual(len((root / "resumed" / "history.jsonl").read_text().splitlines()), 3)

            mismatch = make_server(root / "mismatch", checkpoint)
            mismatch.config.lr *= 2
            with self.assertRaisesRegex(ValueError, "configuration does not match"):
                mismatch.run()

            same_dir = make_server(root / "interrupted", checkpoint)
            with self.assertRaisesRegex(ValueError, "new output directory"):
                same_dir.run()

            wrong_seed = make_server(root / "wrong_seed", checkpoint)
            wrong_seed.run_seed = 456
            with self.assertRaisesRegex(ValueError, "seed does not match"):
                wrong_seed.run()

            wrong_split = make_server(root / "wrong_split", checkpoint)
            wrong_split.client_loaders[0].dataset.indices = [1, 0, 2, 3]
            with self.assertRaisesRegex(ValueError, "split or library versions do not match"):
                wrong_split.run()

    def test_real_unet_checkpoint_is_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = np.arange(32 * 6 * 32, dtype=np.float32).reshape(32, 6, 32)
            for name, base in (("a", 0), ("b", 1)):
                labels = np.zeros(data.shape, dtype=np.uint8) + base
                labels[:, :, 16:] += 1
                np.save(root / f"{name}_data.npy", data)
                np.save(root / f"{name}_labels.npy", labels)
            cfg = RunConfig.from_mapping({
                "dataset": "seismic", "multisurvey": True, "algorithm": "fedhca2",
                "train_seismic": str(root / "a_data.npy"), "train_labels": str(root / "a_labels.npy"),
                "survey_b_seismic": str(root / "b_data.npy"), "survey_b_labels": str(root / "b_labels.npy"),
                "survey_b_offset": 2, "num_clients": 2, "num_classes": 4,
                "num_rounds": 1, "local_epochs": 1, "batch_size": 2, "device": "cpu",
                "num_workers": 0, "output_dir": str(root / "run"),
                "privacy_log_last_only": True, "privacy_save_local": False,
                "save_round_checkpoint": True,
            })
            run(cfg, seeds=[0])
            directory = root / "run" / "seed_0"
            checkpoint = torch.load(directory / "round_checkpoint.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["round"], 1)
            self.assertEqual(checkpoint["seed"], 0)
            self.assertEqual(len(checkpoint["personal"]), 2)
            self.assertIn("train:0", checkpoint["split_fingerprints"])
            self.assertIn("train_seismic", checkpoint["dataset_fingerprints"])
            self.assertTrue((directory / "run_meta_start.json").is_file())

    @unittest.skipUnless(os.environ.get("RUN_GPU_CHECKPOINT_TEST") == "1" and torch.cuda.is_available(),
                         "explicit GPU checkpoint test not requested")
    def test_gpu_resume_matches_within_tolerance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = make_server(root / "full_gpu", device="cuda")
            full.run()
            interrupted = make_server(root / "interrupted_gpu", device="cuda")
            original = interrupted.run_round

            def stop_after_one(index):
                if index == 1:
                    raise RuntimeError("simulated interruption")
                return original(index)

            with patch.object(interrupted, "run_round", side_effect=stop_after_one):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    interrupted.run()
            resumed = make_server(root / "resumed_gpu",
                                  root / "interrupted_gpu/round_checkpoint.pt", device="cuda")
            resumed.run()
            for client in (0, 1):
                actual = resumed._trainer(client).personal_state
                expected = full._trainer(client).personal_state
                for key in actual:
                    self.assertTrue(torch.allclose(actual[key], expected[key], atol=1e-5, rtol=1e-5),
                                    (client, key))


if __name__ == "__main__":
    unittest.main()
