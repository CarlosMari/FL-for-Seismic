"""One-round equivalence gate for the base federated loop."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from fedseismic.config import RunConfig
from fedseismic.data import (
    build_client_loaders,
    compute_client_class_info,
    load_and_normalize,
    partition_noniid,
)
from fedseismic.federated.server import Server
from fedseismic.losses import RecallLossPerSlice
from fedseismic.models import UNet


ROOT = Path(__file__).resolve().parents[1]


def _legacy_train_federated():
    path = ROOT / "train_federated.py"
    if not path.exists():
        pytest.skip("train_federated.py already retired")
    spec = importlib.util.spec_from_file_location("legacy_train_federated_round", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _data_paths():
    paths = [ROOT / "datasets/train/train_seismic.npy", ROOT / "datasets/train/train_labels.npy"]
    if not all(path.exists() for path in paths):
        pytest.skip("seismic training dataset is not available")
    return paths


def test_round_one_global_state_is_bit_exact():
    old = _legacy_train_federated()
    seismic_path, labels_path = _data_paths()
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)
    old_rng = np.random.RandomState(seed)
    old_seismic = old.load_and_normalize(seismic_path)
    labels = np.load(labels_path)
    partitions = old.partition_noniid(old_seismic.shape[1], 20)
    old_loaders = old.build_client_loaders(old_seismic, labels, partitions, 4)
    old_model = old.UNet(in_channels=1, num_classes=6, bilinear=False)
    old_global = deepcopy(old_model.state_dict())
    selected = sorted(old_rng.choice(20, max(1, int(20 * 0.25)), replace=False).tolist())
    old_states = []
    for client in selected:
        local = old.UNet(in_channels=1, num_classes=6, bilinear=False)
        local.load_state_dict(old_global)
        old_states.append(old.local_train(
            local, old_loaders[client], old.RecallLossPerSlice(6), old.DEVICE,
            3, 1e-3, 1e-4,
        ))
    old_result = old.fedavg_aggregate(old_states)

    torch.manual_seed(seed)
    np.random.seed(seed)
    new_rng = np.random.RandomState(seed)
    new_seismic = load_and_normalize(seismic_path)
    new_partitions = partition_noniid(new_seismic.shape[1], 20)
    new_loaders = build_client_loaders(new_seismic, labels, new_partitions, 4)
    new_model = UNet(in_channels=1, num_classes=6, bilinear=False)
    config = RunConfig(
        num_clients=20, num_rounds=1, local_epochs=3, batch_size=4,
        sample_ratio=0.25, loss="recall_slice", agg_strategy="equal",
        device=old.DEVICE,
    )
    new_server = Server(
        model=new_model,
        client_loaders=new_loaders,
        criterion=RecallLossPerSlice(6),
        config=config,
        client_info=compute_client_class_info(labels, new_partitions),
        model_factory=lambda: UNet(in_channels=1, num_classes=6, bilinear=False),
        rng=new_rng,
    )
    new_server.run_round(0)
    new_result = new_server.model.state_dict()

    assert selected == new_server.history[0].selected_clients
    differences = []
    for key in old_result:
        old_tensor = old_result[key].detach().cpu()
        new_tensor = new_result[key].detach().cpu()
        if not torch.equal(old_tensor, new_tensor):
            differences.append((key, (old_tensor - new_tensor).abs().max().item()))
    assert not differences, f"round-1 state differences: {differences[:10]}"
