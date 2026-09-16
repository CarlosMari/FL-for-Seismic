"""Four-layer CNNs used for CIFAR and MedMNIST federated classification."""

import torch.nn as nn


class FedAvgNetCIFAR(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(4096, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv2d_1(x))
        x = self.max_pooling(x)
        x = self.relu(self.conv2d_2(x))
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        return self.classifier(x)


class MedMNISTNet(nn.Module):
    """28x28 four-layer CNN from the OLIVES ``FedAvgNet`` family.

    Same architecture as ``train_tools.models.fedavgnet.MedMNISTNet`` (linear
    3136 = 7x7x64 after two stride-2 pools). BloodMNIST uses 3 channels;
    OrganCMNIST / OrganSMNIST use 1.
    """

    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, get_features=False):
        x = self.relu(self.conv2d_1(x))
        x = self.max_pooling(x)
        x = self.relu(self.conv2d_2(x))
        x = self.max_pooling(x)
        x = self.flatten(x)
        features = self.relu(self.linear_1(x))
        logits = self.classifier(features)
        if get_features:
            return logits, features
        return logits
