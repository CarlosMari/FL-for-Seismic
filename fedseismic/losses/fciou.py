"""Focal class-based IoU loss."""

import torch.nn as nn
import torch.nn.functional as F


class FCIoULoss(nn.Module):
    def __init__(self, gamma=2.0, smooth=1.0):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        union = probs.sum(dim=dims) + one_hot.sum(dim=dims) - intersection
        iou_per_class = (intersection + self.smooth) / (union + self.smooth)
        loss_per_class = (1.0 - iou_per_class) ** self.gamma * (1.0 - iou_per_class)
        return loss_per_class.mean()
