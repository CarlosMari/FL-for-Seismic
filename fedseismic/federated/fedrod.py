"""FedRoD: a shared generic head plus a personal head that stays local.

Chen and Chao, ICLR 2022. The generic classifier is aggregated. The personal
classifier is a residual on the same features and is never uploaded. Training
uses balanced softmax, adding the client's log class counts to the summed
logits. The deployed model is the sum of the two heads.
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from .client import ClientTrainer, _logits, class_frequency


class FedRoDClientTrainer(ClientTrainer):
    """Train body and generic head. Keep the personal residual on the client."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.personal_head = None
        self.personal_state = None
        self._upload_state = None

    def _ensure_head(self):
        classifier = self.model.classifier
        if self.personal_head is None:
            self.personal_head = nn.Linear(
                classifier.in_features, classifier.out_features, device=self.device,
            )
            nn.init.zeros_(self.personal_head.weight)
            nn.init.zeros_(self.personal_head.bias)
        self.personal_head.to(self.device)

    def _optimizer(self):
        params = list(self.model.parameters()) + list(self.personal_head.parameters())
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay,
            )
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        raise ValueError(f"unknown optimizer {self.optimizer_name!r}")

    def download(self, state_dict):
        super().download(state_dict)
        return self

    def train(self, global_state=None):
        self.model.train()
        self.model.to(self.device)
        self._ensure_head()
        self.optimizer = self._optimizer()
        counts = class_frequency(self.loader, self.num_classes)
        log_prior = torch.as_tensor(
            np.log(np.clip(counts, 1e-8, None)), device=self.device, dtype=torch.float,
        )
        for _ in range(self.local_epochs):
            for images, targets, _ in self.loader:
                images = images.to(self.device, dtype=torch.float)
                targets = targets.to(self.device, dtype=torch.long)
                generic, features = self.model(images, get_features=True)
                generic = _logits(generic)
                scored = generic + self.personal_head(features) + log_prior
                loss = self.criterion(scored, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        generic_state = deepcopy(self.model.state_dict())
        deployed = deepcopy(generic_state)
        deployed["classifier.weight"] = (
            generic_state["classifier.weight"] + self.personal_head.weight.detach()
        )
        deployed["classifier.bias"] = (
            generic_state["classifier.bias"] + self.personal_head.bias.detach()
        )
        self._upload_state = generic_state
        self.personal_state = deployed
        return self

    def upload(self):
        if self._upload_state is None:
            return super().upload()
        return deepcopy(self._upload_state)
