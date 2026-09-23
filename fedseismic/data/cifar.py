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


def partition_presence(targets, num_clients, rng, n_rare=2, absent_fraction=0.5,
                       rate_low=0.01, rate_high=0.05, alpha=1.0):
    """Most classes are a mild Dirichlet split. The rarest classes are not.

    Half the clients, by default, receive none of a rare class. The others
    receive enough that the class is between ``rate_low`` and ``rate_high``
    of that client's data. Leftover rare images are left out on purpose.
    Returns ``(partitions, rare_class_indices)``.
    """
    targets = np.asarray(targets).reshape(-1)
    counts = np.bincount(targets)
    rare = [int(index) for index in np.argsort(counts)[:n_rare]]
    base_ids = np.where(~np.isin(targets, rare))[0]
    relative = partition_dirichlet(targets[base_ids], num_clients, alpha, rng)
    parts = [[int(base_ids[index]) for index in client] for client in relative]
    for class_index in rare:
        class_ids = rng.permutation(np.where(targets == class_index)[0])
        n_absent = int(round(absent_fraction * num_clients))
        absent = set(int(client) for client in rng.choice(num_clients, size=n_absent, replace=False))
        cursor = 0
        for client in rng.permutation(num_clients):
            if int(client) in absent or cursor >= len(class_ids):
                continue
            rate = float(rng.uniform(rate_low, rate_high))
            want = max(1, int(round(rate / max(1.0 - rate, 1e-6) * max(len(parts[int(client)]), 1))))
            take = class_ids[cursor:cursor + want]
            cursor += len(take)
            parts[int(client)].extend(int(index) for index in take)
    for client, indices in enumerate(parts):
        if len(indices) < 2:
            donor = max(range(num_clients), key=lambda other: len(parts[other]))
            move = parts[donor][:2]
            parts[donor] = parts[donor][2:]
            parts[client] = list(indices) + move
    return [sorted(set(indices)) for indices in parts], tuple(rare)


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
