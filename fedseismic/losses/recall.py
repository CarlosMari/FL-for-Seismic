"""Recall-weighted losses for rare seismic facies."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrosslineRecallLoss(nn.Module):
    def __init__(self, rare_classes, smooth=1.0):
        super().__init__()
        self.rare_classes = rare_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        penalties = []
        for b in range(targets.shape[0]):
            tgt_b = targets[b]
            _ = preds[b]
            prob_b = probs[b]
            slice_terms = []
            for c in self.rare_classes:
                gt_mask = (tgt_b == c)
                if gt_mask.sum() == 0:
                    continue
                recall = (prob_b[c] * gt_mask.float()).sum() / (gt_mask.float().sum() + self.smooth)
                slice_terms.append(1.0 - recall)
            if slice_terms:
                penalties.append(torch.stack(slice_terms).mean())
        if not penalties:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return torch.stack(penalties).mean()


class CompoundCrosslineRecallLoss(nn.Module):
    def __init__(self, rare_classes, lambda_crl=1.0, gamma=2.0, alpha=None):
        super().__init__()
        from .dice import FocalDiceLoss
        self.base = FocalDiceLoss(gamma=gamma, alpha=alpha)
        self.crl = CrosslineRecallLoss(rare_classes=rare_classes)
        self.lambda_crl = lambda_crl

    def forward(self, logits, targets):
        return self.base(logits, targets) + self.lambda_crl * self.crl(logits, targets)


class RecallLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        preds = logits.argmax(dim=1)
        w = torch.zeros(self.num_classes, device=logits.device)
        for c in range(self.num_classes):
            gt_c = (targets == c)
            n_gt = gt_c.sum().float()
            if n_gt < 1:
                w[c] = 0.0
                continue
            tp = ((preds == c) & gt_c).sum().float()
            fn = n_gt - tp
            w[c] = fn / (fn + tp + self.smooth)
        return F.cross_entropy(logits, targets, weight=w.detach(), reduction="mean")


class RecallLossPerSlice(nn.Module):
    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        preds = logits.argmax(dim=1)
        losses = []
        for b in range(targets.shape[0]):
            tgt_b = targets[b]
            prd_b = preds[b]
            w = torch.zeros(self.num_classes, device=logits.device)
            for c in range(self.num_classes):
                gt_c = (tgt_b == c)
                n_gt = gt_c.sum().float()
                if n_gt < 1:
                    w[c] = 0.0
                    continue
                tp = ((prd_b == c) & gt_c).sum().float()
                fn = n_gt - tp
                w[c] = fn / (fn + tp + self.smooth)
            losses.append(F.cross_entropy(
                logits[b:b + 1], targets[b:b + 1], weight=w.detach(), reduction="mean",
            ))
        return torch.stack(losses).mean()
