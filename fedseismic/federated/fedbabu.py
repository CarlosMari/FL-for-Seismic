"""FedBABU: aggregate the body, personalize the classifier once at the end.

Oh, Park, and Kim, ICLR 2022. During federation the classifier is frozen at
its shared initialization, so the upload is a body trained under that fixed
head. After the last round each client trains only its classifier on local
data. That head is the deployed model and is not uploaded.
"""

from copy import deepcopy

import torch

from .client import ClientTrainer
from .fedper import personalization_keys


class FedBABUClientTrainer(ClientTrainer):
    """Train the body against a frozen head. ``personalize`` fits the head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.personal_state = None

    def _make_optimizer(self):
        params = [param for param in self.model.parameters() if param.requires_grad]
        if self.optimizer_name == "sgd":
            return self._sgd(params)
        if self.optimizer_name == "adamw":
            return self._adamw(params)
        raise ValueError(f"unknown optimizer {self.optimizer_name!r}")

    def _sgd(self, params):
        return torch.optim.SGD(
            params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay,
        )

    def _adamw(self, params):
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def _set_trainable(self, head_only):
        head = set(personalization_keys(self.model.state_dict()))
        for name, param in self.model.named_parameters():
            param.requires_grad = (name in head) if head_only else (name not in head)

    def _restore_trainable(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def train(self, global_state=None):
        self._set_trainable(head_only=False)
        try:
            super().train(global_state=global_state)
        finally:
            self._restore_trainable()
        return self

    def personalize(self):
        """Fit the classifier on local data. The body stays at the global model."""
        self._set_trainable(head_only=True)
        try:
            super().train()
        finally:
            self._restore_trainable()
        self.personal_state = deepcopy(self.model.state_dict())
        return self
