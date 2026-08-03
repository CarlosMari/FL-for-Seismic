import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fedseismic.eval.metrics import evaluate


class FixedModel(torch.nn.Module):
    def forward(self, images):
        logits = torch.zeros(images.shape[0], 2, images.shape[2], images.shape[3])
        logits[:, 0] = 1
        return logits, torch.zeros_like(images)


def test_evaluate_returns_macro_and_per_class_iou():
    images = torch.zeros(2, 1, 2, 2)
    targets = torch.zeros(2, 2, 2, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, targets, torch.arange(2)))
    labels = np.zeros((2, 2, 2), dtype=np.int64)
    miou, per_class = evaluate(FixedModel(), loader, labels)
    assert per_class.shape == (1,)
    assert miou == 1.0
