"""Client class metadata must describe the exact selected training sections."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fedseismic.config import RunConfig
from fedseismic.experiment import build_federated_run


class TestMultisurveySubset(unittest.TestCase):
    def test_client_info_uses_selected_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seismic = np.arange(16 * 20 * 16, dtype=np.float32).reshape(16, 20, 16)
            labels = np.zeros_like(seismic, dtype=np.uint8)
            labels[:, 8:, :] = 1  # Class 1 is outside the selected train subset.
            for survey in ("a", "b"):
                np.save(root / f"{survey}_seismic.npy", seismic)
                np.save(root / f"{survey}_labels.npy", labels)
            cfg = RunConfig.from_mapping({
                "dataset": "seismic", "multisurvey": True,
                "train_seismic": str(root / "a_seismic.npy"),
                "train_labels": str(root / "a_labels.npy"),
                "survey_b_seismic": str(root / "b_seismic.npy"),
                "survey_b_labels": str(root / "b_labels.npy"),
                "survey_b_offset": 2, "num_classes": 4, "num_clients": 2,
                "local_test_ratio": 0.2, "survey_train_subset": 8,
                "batch_size": 2, "num_workers": 0, "device": "cpu",
            })
            data = build_federated_run(cfg, 0)
            for index, expected in enumerate(({0}, {2})):
                dataset = data.loaders[index].dataset
                actual = set(np.unique(dataset.label[:, dataset.indices, :]).tolist())
                self.assertEqual(actual, expected)
                self.assertEqual(data.client_info[index]["has_classes"], expected)
                self.assertEqual(set(np.flatnonzero(data.client_info[index]["class_fracs"])), expected)


if __name__ == "__main__":
    unittest.main()
