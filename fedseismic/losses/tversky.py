"""Asymmetric Tversky and unified focal losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .focal import FocalLoss


class AsymmetricTverskyLoss(nn.Module):
    def __init__(self, alpha_per_class, beta_per_class, gamma=1.0, smooth=1.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha_per_class, dtype=torch.float))
        self.register_buffer("beta", torch.tensor(beta_per_class, dtype=torch.float))
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        tp = (probs * one_hot).sum(dim=dims)
        fp = (probs * (1 - one_hot)).sum(dim=dims)
        fn = ((1 - probs) * one_hot).sum(dim=dims)
        alpha = self.alpha.to(logits.device)
        beta = self.beta.to(logits.device)
        t_idx = (tp + self.smooth) / (tp + alpha * fp + beta * fn + self.smooth)
        loss = (1.0 - t_idx) ** (1.0 / self.gamma) if self.gamma != 1.0 else (1.0 - t_idx)
        return loss.mean()


class UnifiedFocalLoss(nn.Module):
    def __init__(self, alpha_per_class, beta_per_class, gamma_ce=2.0,
                 gamma_t=2.0, lambda_ce=0.5, smooth=1.0):
        super().__init__()
        self.focal_ce = FocalLoss(gamma=gamma_ce)
        self.focal_tv = AsymmetricTverskyLoss(
            alpha_per_class=alpha_per_class, beta_per_class=beta_per_class,
            gamma=gamma_t, smooth=smooth,
        )
        self.lambda_ce = lambda_ce

    def forward(self, logits, targets):
        return (self.lambda_ce * self.focal_ce(logits, targets)
                + (1.0 - self.lambda_ce) * self.focal_tv(logits, targets))
