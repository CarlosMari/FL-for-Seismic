import torch.utils.data as data
from torchvision.datasets import CIFAR10

import numpy as np


class CIFAR10_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, mode='tr', transform=None, download=True):
        root = ('/').join((root.split('/'))[:-1])
        self.root = root
        self.dataidxs = dataidxs
        if mode == 'tr':
            train = True
        else:
            train = False
        self.train = train
        self.transform = transform
        self.download = download
        self.num_classes = 10
        self.mode = mode
        self.data, self.targets = self._build_truncated_dataset()

    def _build_truncated_dataset(self):
        base_dataset = CIFAR10(
            self.root, self.train, self.transform, None, self.download
        )

        data = base_dataset.data
        targets = np.array(base_dataset.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, targets, index

    def __len__(self):
        return len(self.data)
