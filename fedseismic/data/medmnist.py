"""MedMNIST loaders for the FedKPer classification replica."""

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToPILImage, ToTensor

from .cifar import IndexedCIFAR10

MEDMNIST_SPECS = {
    "bloodmnist": {"n_classes": 8, "n_channels": 3},
    "organcmnist": {"n_classes": 11, "n_channels": 1},
    "organsmnist": {"n_classes": 11, "n_channels": 1},
}

_CLASS_BY_NAME = {
    "bloodmnist": "BloodMNIST",
    "organcmnist": "OrganCMNIST",
    "organsmnist": "OrganSMNIST",
}


def _transforms(n_channels):
    mean = tuple([0.5] * n_channels)
    std = tuple([0.5] * n_channels)
    return Compose([ToPILImage(), ToTensor(), Normalize(mean, std)])


def _as_uint8_images(images, n_channels):
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[..., None]
    if images.shape[-1] == 1 and n_channels == 1:
        images = images[..., 0]
    return images


def load_medmnist(name, root, download=True):
    """Return train/test ``(images, targets)`` for a 2D MedMNIST split."""
    name = name.lower()
    if name not in MEDMNIST_SPECS:
        raise ValueError(f"unknown MedMNIST dataset {name!r}")
    try:
        import medmnist
    except ImportError as exc:
        raise ImportError("MedMNIST runs need the medmnist package") from exc
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dataset_cls = getattr(medmnist, _CLASS_BY_NAME[name])
    train = dataset_cls(split="train", download=download, root=str(root))
    test = dataset_cls(split="test", download=download, root=str(root))
    spec = MEDMNIST_SPECS[name]
    train_images = _as_uint8_images(train.imgs, spec["n_channels"])
    test_images = _as_uint8_images(test.imgs, spec["n_channels"])
    train_targets = np.asarray(train.labels).reshape(-1)
    test_targets = np.asarray(test.labels).reshape(-1)
    return train_images, train_targets, test_images, test_targets, spec


def build_medmnist_loaders(images, targets, partitions, batch_size, n_channels,
                           train=True, num_workers=0, pin_memory=False):
    transform = _transforms(n_channels)
    loaders = []
    for indices in partitions:
        dataset = IndexedCIFAR10(images, targets, indices, transform)
        loaders.append(DataLoader(
            dataset, batch_size=batch_size, shuffle=train,
            num_workers=num_workers, pin_memory=pin_memory,
        ))
    return loaders


def build_medmnist_test_loader(images, targets, batch_size, n_channels,
                               num_workers=0, pin_memory=False):
    dataset = IndexedCIFAR10(
        images, targets, list(range(len(targets))), _transforms(n_channels),
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
