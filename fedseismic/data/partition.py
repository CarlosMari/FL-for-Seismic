"""Geographic and IID client partitions plus class statistics."""

import numpy as np
from torch.utils.data import DataLoader

from .seismic import InlineLoader, _to_tensor

NUM_CLASSES = 6
RARE_CLASSES = (4, 5)


def partition_noniid(num_crosslines, num_clients):
    chunk_size = num_crosslines // num_clients
    partitions = []
    for client in range(num_clients):
        start = client * chunk_size
        end = (client + 1) * chunk_size if client < num_clients - 1 else num_crosslines
        partitions.append(list(range(start, end)))
    return partitions


def partition_iid(num_crosslines, num_clients, rng):
    all_idxs = np.arange(num_crosslines)
    rng.shuffle(all_idxs)
    chunk_size = num_crosslines // num_clients
    partitions = []
    for client in range(num_clients):
        start = client * chunk_size
        end = (client + 1) * chunk_size if client < num_clients - 1 else num_crosslines
        partitions.append(sorted(all_idxs[start:end].tolist()))
    return partitions


def split_client_local_test(partitions, local_test_ratio=0.2, rng=None):
    """Hold out part of each client's own indices.

    With ``rng is None`` (seismic), take the contiguous tail of the sorted
    chunk. With an RNG (CIFAR), draw a random subset so the holdout is not
    an artifact of image index order.
    """
    if not 0 <= local_test_ratio < 1:
        raise ValueError("local_test_ratio must be in [0, 1)")
    train_partitions = []
    test_partitions = []
    for idxs in partitions:
        ordered = sorted(idxs)
        if local_test_ratio == 0 or len(ordered) < 2:
            train_partitions.append(ordered)
            test_partitions.append([])
            continue
        n_test = max(1, int(round(len(ordered) * local_test_ratio)))
        n_test = min(n_test, len(ordered) - 1)
        if rng is None:
            test_idxs = ordered[-n_test:]
        else:
            chosen = rng.choice(len(ordered), size=n_test, replace=False)
            test_idxs = sorted(ordered[int(i)] for i in chosen)
        test_set = set(test_idxs)
        train_partitions.append([idx for idx in ordered if idx not in test_set])
        test_partitions.append(test_idxs)
    return train_partitions, test_partitions


def build_client_loaders(train_seismic, train_labels, partitions, batch_size,
                         num_workers=2, pin_memory=True, shuffle=True,
                         train_status=True):
    loaders = []
    for crossline_idxs in partitions:
        dataset = InlineLoader(
            seismic_cube=train_seismic,
            label_cube=train_labels,
            inline_inds=crossline_idxs,
            train_status=train_status,
            transform=_to_tensor,
        )
        loaders.append(DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        ))
    return loaders


def compute_client_class_info(train_labels, partitions, num_classes=NUM_CLASSES,
                              rare_classes=RARE_CLASSES):
    client_info = []
    labels = np.asarray(train_labels)
    for idxs in partitions:
        if labels.ndim > 1:
            client_labels = labels[:, idxs, :].flatten()
        else:
            client_labels = labels[idxs]
        unique_classes = set(np.unique(client_labels).tolist())
        class_counts = dict(zip(*np.unique(client_labels, return_counts=True)))
        total = max(int(client_labels.size), 1)
        rare_fraction = sum(class_counts.get(rc, 0) for rc in rare_classes) / total
        class_fracs = np.zeros(num_classes, dtype=np.float64)
        for class_idx in range(num_classes):
            class_fracs[class_idx] = class_counts.get(class_idx, 0) / max(total, 1)
        client_info.append({
            "num_classes": len(unique_classes),
            "has_classes": unique_classes,
            "rare_fraction": rare_fraction,
            "class_fracs": class_fracs,
        })
    return client_info
