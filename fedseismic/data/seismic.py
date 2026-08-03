"""Dataset primitives for Parihaka-style seismic cubes."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class InlineLoader(Dataset):
    """Expose crossline sections from an ``(inline, crossline, depth)`` cube.

    The historical API calls the indices ``inline_inds`` even though they
    index axis 1.  That name is retained for checkpoint and loader parity.
    """

    def __init__(self, seismic_cube, label_cube, inline_inds, train_status=True,
                 transform=None, preprocessing=None):
        self.seismic = seismic_cube
        self.label = label_cube
        self.indices = inline_inds
        self.train_status = train_status
        self.transform = transform
        self.preprocessing = preprocessing

    def __getitem__(self, index):
        inline_num = self.indices[index]
        section = self.seismic[:, inline_num, :].T
        label_section = self.label[:, inline_num, :].T
        if self.preprocessing is not None:
            processed = self.preprocessing(image=section, mask=label_section)
            section, label_section = processed["image"], processed["mask"]
        if self.transform is not None:
            section = self.transform(section)
        else:
            section = torch.as_tensor(section, dtype=torch.float32).unsqueeze(0)
        return section, torch.as_tensor(label_section, dtype=torch.long), index

    def __len__(self):
        return len(self.indices)


def load_and_normalize(path):
    arr = np.load(Path(path))
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def _to_tensor(section):
    tensor = torch.as_tensor(section, dtype=torch.float32)
    return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor


def build_test_loader(seismic_path, labels_path, batch_size=1, num_workers=2,
                      pin_memory=True):
    seismic = load_and_normalize(seismic_path)
    labels = np.load(labels_path)
    dataset = InlineLoader(
        seismic_cube=seismic,
        label_cube=labels,
        inline_inds=list(range(seismic.shape[1])),
        train_status=False,
        transform=_to_tensor,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    ), labels
