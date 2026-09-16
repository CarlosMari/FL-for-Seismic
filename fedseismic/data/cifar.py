"""CIFAR-10 loaders, Dirichlet/IID partitions, and per-client holdouts."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToPILImage, ToTensor

CIFAR_MEAN = (0.49139968, 0.48215827, 0.44653124)
CIFAR_STD = (0.24703233, 0.24348505, 0.26158768)


class IndexedCIFAR10(Dataset):
    """Return ``(image, label, index)`` so trainers share the seismic batch contract."""

    def __init__(self, images, targets, indices, transform):
        self.images = images
        self.targets = np.asarray(targets)
        self.indices = list(indices)
        self.transform = transform

    def __getitem__(self, index):
        source = self.indices[index]
        image = self.images[source]
        if self.transform is not None:
            image = self.transform(image)
        target = int(self.targets[source])
        return image, torch.tensor(target, dtype=torch.long), index

    def __len__(self):
        return len(self.indices)


def _transforms(train):
    normalize = Normalize(CIFAR_MEAN, CIFAR_STD)
    if train:
        return Compose([
            ToPILImage(), RandomCrop(32, padding=4), RandomHorizontalFlip(),
            ToTensor(), normalize,
        ])
    return Compose([ToPILImage(), ToTensor(), normalize])


def load_cifar10(root, download=True):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train = CIFAR10(root=root, train=True, download=download)
    test = CIFAR10(root=root, train=False, download=download)
    return train.data, np.asarray(train.targets), test.data, np.asarray(test.targets)


def partition_dirichlet(targets, num_clients, alpha, rng):
    targets = np.asarray(targets)
    num_classes = int(targets.max()) + 1
    client_indices = [[] for _ in range(num_clients)]
    for class_index in range(num_classes):
        class_ids = rng.permutation(np.where(targets == class_index)[0])
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        splits = (np.cumsum(proportions) * len(class_ids)).astype(int)[:-1]
        for client, part in enumerate(np.split(class_ids, splits)):
            client_indices[client].extend(part.tolist())
    return [sorted(indices) for indices in client_indices]


def partition_iid_indices(num_samples, num_clients, rng):
    order = rng.permutation(num_samples)
    chunk = num_samples // num_clients
    partitions = []
    for client in range(num_clients):
        start = client * chunk
        end = (client + 1) * chunk if client < num_clients - 1 else num_samples
        partitions.append(sorted(order[start:end].tolist()))
    return partitions


def build_cifar_loaders(images, targets, partitions, batch_size, train=True,
                        num_workers=0, pin_memory=False):
    transform = _transforms(train)
    loaders = []
    for indices in partitions:
        dataset = IndexedCIFAR10(images, targets, indices, transform)
        loaders.append(DataLoader(
            dataset, batch_size=batch_size, shuffle=train,
            num_workers=num_workers, pin_memory=pin_memory,
        ))
    return loaders


def build_cifar_test_loader(images, targets, batch_size, num_workers=0, pin_memory=False):
    dataset = IndexedCIFAR10(images, targets, list(range(len(targets))), _transforms(False))
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
