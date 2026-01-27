import torch
import torch.nn as nn


class FedAvgNetMNIST(torch.nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(FedAvgNetMNIST, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, get_features=False):
        if x.ndim < 4:
            x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.relu(self.linear_1(x))
        x = self.classifier(z)
        if get_features:
            return x, z

        else:
            return x

class olivesNet(torch.nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(olivesNet, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(200704, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, get_features=False):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.relu(self.linear_1(x))
        x = self.classifier(z)

        if get_features:
            return x, z

        else:
            return x

class MedMNISTNet(torch.nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(MedMNISTNet, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, get_features=False):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.relu(self.linear_1(x))
        x = self.classifier(z)

        if get_features:
            return x, z

        else:
            return x

class FedAvgNetCIFAR(torch.nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(FedAvgNetCIFAR, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(4096, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, get_features=False):
        x = self.conv2d_1(x)
        x = self.relu(x)
        y = self.max_pooling(x)
        x = self.conv2d_2(y)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.relu(self.linear_1(x))
        x = self.classifier(z)

        if get_features:
            return x, z

        else:
            return x


class FedAvgNetTiny(torch.nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(FedAvgNetTiny, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(16384, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, get_features=False):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.relu(self.linear_1(x))
        x = self.classifier(z)

        if get_features:
            return x, z

        else:
            return x


class FedAvgNetFashion(torch.nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(FedAvgNetFashion, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(1568, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, get_features=False):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.relu(self.linear_1(x))
        x = self.classifier(z)

        if get_features:
            return x, z

        else:
            return x
