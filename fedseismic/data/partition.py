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


def build_client_loaders(train_seismic, train_labels, partitions, batch_size,
                         num_workers=2, pin_memory=True):
    loaders = []
    for crossline_idxs in partitions:
        dataset = InlineLoader(
            seismic_cube=train_seismic,
            label_cube=train_labels,
            inline_inds=crossline_idxs,
            train_status=True,
            transform=_to_tensor,
        )
        loaders.append(DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        ))
    return loaders


def compute_client_class_info(train_labels, partitions, num_classes=NUM_CLASSES,
                              rare_classes=RARE_CLASSES):
    client_info = []
    for idxs in partitions:
        client_labels = train_labels[:, idxs, :].flatten()
        unique_classes = set(np.unique(client_labels).tolist())
        class_counts = dict(zip(*np.unique(client_labels, return_counts=True)))
        total = client_labels.size
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
