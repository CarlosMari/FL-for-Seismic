"""Round-1 gates for the ported FedProx, FedBN, and FedVLS variants."""

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
from fedseismic.losses import DiceLoss, FocalDiceLoss
from fedseismic.models import UNet


ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def _load_legacy(name):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} already retired")
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paths():
    paths = [ROOT / "datasets/train/train_seismic.npy", ROOT / "datasets/train/train_labels.npy"]
    if not all(path.exists() for path in paths):
        pytest.skip("seismic training dataset is not available")
    return paths


def _state_difference(old_state, new_state):
    differences = []
    for key in old_state:
        old_tensor = old_state[key].detach().cpu()
        new_tensor = new_state[key].detach().cpu()
        if not torch.equal(old_tensor, new_tensor):
            differences.append((key, (old_tensor - new_tensor).abs().max().item()))
    return differences


def _run_variant(variant, legacy_name, monkeypatch):
    monkeypatch.setenv("UNET_NORM", "batch")
    old = _load_legacy(legacy_name)
    seismic_path, labels_path = _paths()
    labels = np.load(labels_path)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    old_seismic = old.load_and_normalize(seismic_path)
    partitions = old.partition_noniid(old_seismic.shape[1], 20)
    old_loaders = old.build_client_loaders(old_seismic, labels, partitions, 4)
    old_model = old.UNet(in_channels=1, num_classes=6, bilinear=False)
    global_state = deepcopy(old_model.state_dict())
    old_states = []

    if variant == "fedbn":
        bn_keys = [key for key in global_state if old.is_bn_key(key)]
        client_bn_states = [None] * 20
    if variant == "fedvls":
        client_freqs = []
        client_vacant = []
        for indexes in partitions:
            values, counts = np.unique(labels[:, indexes, :].flatten(), return_counts=True)
            frequency = np.zeros(6, dtype=np.float32)
            for value, count in zip(values, counts):
                frequency[int(value)] = count / counts.sum()
            client_freqs.append(frequency)
            client_vacant.append(frequency == 0)

    frozen = None
    if variant == "fedvls":
        frozen = old.UNet(in_channels=1, num_classes=6, bilinear=False)
        frozen.load_state_dict(global_state)
    for client in range(20):
        local = old.UNet(in_channels=1, num_classes=6, bilinear=False)
        local.load_state_dict(global_state, strict=False)
        if variant == "fedbn" and client_bn_states[client] is not None:
            local_state = local.state_dict()
            for key in bn_keys:
                local_state[key] = client_bn_states[client][key]
            local.load_state_dict(local_state)
        if variant == "fedprox":
            updated = old.local_train_fedprox(
                local, old_loaders[client], old.FocalDiceLoss(), old.DEVICE, 3,
                1e-3, 1e-4, [parameter.detach().clone() for parameter in old_model.parameters()], 0.01,
            )
        elif variant == "fedbn":
            updated = old.local_train(
                local, old_loaders[client], old.FocalDiceLoss(), old.DEVICE, 3, 1e-3, 1e-4,
            )
            client_bn_states[client] = {key: updated[key].clone() for key in bn_keys}
        else:
            updated = old.local_train_fedvls(
                local, frozen, old_loaders[client], old.DEVICE, 3, 1e-3, 1e-4,
                client_freqs[client], client_vacant[client], 0.1,
            )
        old_states.append(updated)
    if variant == "fedbn":
        old_result = old.fedbn_aggregate(old_states)
    else:
        old_result = old.fedavg_aggregate(old_states)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    new_seismic = load_and_normalize(seismic_path)
    new_partitions = partition_noniid(new_seismic.shape[1], 20)
    new_loaders = build_client_loaders(new_seismic, labels, new_partitions, 4)
    new_model = UNet(in_channels=1, num_classes=6, bilinear=False)
    config = RunConfig(
        num_clients=20, num_rounds=1, local_epochs=3, batch_size=4,
        sample_ratio=1.0, algorithm=variant, device=old.DEVICE,
    )
    criterion = DiceLoss() if variant == "fedvls" else FocalDiceLoss()
    new_server = Server(
        model=new_model, client_loaders=new_loaders, criterion=criterion,
        config=config, client_info=compute_client_class_info(labels, new_partitions),
        model_factory=lambda: UNet(in_channels=1, num_classes=6, bilinear=False),
        rng=np.random.RandomState(SEED),
    )
    new_server.run_round(0)
    differences = _state_difference(old_result, new_server.model.state_dict())
    assert not differences, f"{variant} round-1 state differences: {differences[:10]}"


def test_fedprox_round_one_is_bit_exact(monkeypatch):
    _run_variant("fedprox", "train_fedprox.py", monkeypatch)


def test_fedbn_round_one_is_bit_exact(monkeypatch):
    _run_variant("fedbn", "train_fedbn.py", monkeypatch)


def test_fedvls_round_one_is_bit_exact(monkeypatch):
    _run_variant("fedvls", "train_fedvls.py", monkeypatch)
