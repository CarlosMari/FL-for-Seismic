"""FedKPer local trainer: adaptive knowledge personalization.

Fowler & AlRegib, ICIP 2026 (arXiv:2605.00698). Aggregation for this method
lives in ``aggregation.FedKPerAgg``; this module only overrides local training.
"""

from copy import deepcopy

import torch
import torch.nn.functional as F

from .client import ClientTrainer, _logits


class FedKPerClientTrainer(ClientTrainer):
    """Pixel CE plus reliability-weighted KD against the frozen global model."""

    def __init__(self, *args, lambda_cap=10.0, grad_clip=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_cap = lambda_cap
        self.grad_clip = grad_clip

    def train(self, global_state=None):
        if global_state is None:
            global_state = deepcopy(self.model.state_dict())
        global_model = deepcopy(self.model).to(self.device)
        global_model.load_state_dict(global_state)
        global_model.eval()
        self.model.train()
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        for _ in range(self.local_epochs):
            for images, targets, _ in self.loader:
                images = images.to(self.device, dtype=torch.float)
                targets = targets.to(self.device, dtype=torch.long)
                logits = _logits(self.model(images))
                with torch.no_grad():
                    global_logits = _logits(global_model(images))
                    teacher_ce = F.cross_entropy(global_logits, targets)
                    lam = min(self.lambda_cap, 1.0 / float(teacher_ce.clamp(min=1e-8)))
                loss = F.cross_entropy(logits, targets) + lam * _distillation_kl(logits, global_logits)
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip is not None and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
        return self


def _distillation_kl(student_logits, teacher_logits):
    """Mean KL(teacher || student) over the class dimension."""
    log_student = F.log_softmax(student_logits, dim=1)
    teacher = F.softmax(teacher_logits, dim=1)
    if student_logits.ndim == 4:
        classes = student_logits.shape[1]
        log_student = log_student.permute(0, 2, 3, 1).reshape(-1, classes)
        teacher = teacher.permute(0, 2, 3, 1).reshape(-1, classes)
    return F.kl_div(log_student, teacher, reduction="batchmean")
