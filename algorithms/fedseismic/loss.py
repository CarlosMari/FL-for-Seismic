import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["weighted_loss"]


class weighted_loss(nn.Module):

    def __init__(self, beta):
        super(weighted_loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.beta = beta

    def forward(self, logits, targets, dg_logits):

        ce_loss = self.CE(logits, targets)

        global_ce = self.CE(dg_logits, targets)

        kl_loss = self.kl_loss(logits, dg_logits)
        loss = ce_loss + (1/global_ce) * kl_loss

        return loss

    def kl_loss(self, logits, dg_logits):

        pred_probs = F.log_softmax(logits, dim=1)

        with torch.no_grad():
            dg_probs = torch.softmax(dg_logits, dim=1)

        loss = self.KLDiv(pred_probs, dg_probs)

        return loss