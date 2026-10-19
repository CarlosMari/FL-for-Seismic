"""FedPer + FedBN combined: shared body, local head, local BatchNorm.

Setup B for multi-silo FL. The final classifier (``outc``) and the
BatchNorm statistics/affines stay on the client and are never aggregated;
only the shared body is averaged. Each client deploys body + its own head +
its own BN.

This removes two couplings a shared 12-class head + averaged BN imposes:
the cross-survey negative-logit gradient into the body (loss coupling) and
the cross-survey BN statistic mixing. Per-survey/per-client heads learn
their own weights on their own data — the mechanism the frozen-body probe
showed recovers accuracy.
"""

from copy import deepcopy

import numpy as np
import torch

from .client import _dataset_labels, _is_bn_key, _logits
from .fedper import FedPerClientTrainer, personalization_keys


class FedPerBNClientTrainer(FedPerClientTrainer):
    """Keep the head and BatchNorm local; aggregate only the body.

    Same upload trick as FedPer (the global head is written back into the
    upload so aggregation leaves it unchanged) extended to BN: the global BN
    is written back too, so the shared model's BN is a dummy each client
    overrides on download. The deployed ``personal_state`` keeps the client's
    own head and BN.
    """

    def _local_keys(self, state_dict):
        return set(personalization_keys(state_dict, self.personal_prefix)) | {
            key for key in state_dict if _is_bn_key(key)
        }

    def _own_classes(self):
        labels = _dataset_labels(getattr(self.loader, "dataset", None))
        if labels is None:
            return None
        present = np.unique(np.asarray(labels).reshape(-1).astype(int))
        return torch.as_tensor(present, dtype=torch.long)

    def train(self, global_state=None):
        """Train with cross-entropy masked to the classes this client holds.

        Masking the softmax to own classes is mathematically the same as a
        per-survey head (only own-class rows get gradient) and removes the
        cross-survey negative-logit gradient into the shared body.
        """
        if global_state is None:
            global_state = deepcopy(self.model.state_dict())
        self.model.train()
        self.model.to(self.device)
        self.optimizer = self._make_optimizer()
        own = self._own_classes()
        own = None if own is None else own.to(self.device)
        for _ in range(max(int(self.local_epochs), 1)):
            for images, targets, _ in self.loader:
                images = images.to(self.device, dtype=torch.float)
                targets = targets.to(self.device, dtype=torch.long)
                logits = _logits(self.model(images))
                if own is not None and own.numel() < logits.shape[1]:
                    logits = logits.index_select(1, own)
                    remap = torch.searchsorted(own, targets)
                    targets = remap
                loss = torch.nn.functional.cross_entropy(logits, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        # Rebuild personal_state (body + own head + own BN) and the upload
        # (body + global head + global BN) so only the body is aggregated.
        trained = deepcopy(self.model.state_dict())
        self.personal_state = trained
        uploaded = deepcopy(trained)
        for key in self._local_keys(trained):
            uploaded[key] = global_state[key].detach().clone()
        self._upload_state = uploaded
        self.model.load_state_dict(uploaded)
        return self

    def download(self, state_dict):
        # Overlay own head + own BN on top of the shared body.
        super(FedPerClientTrainer, self).download(state_dict)
        if self.personal_state is None:
            return self
        overlay = {
            key: self.personal_state[key]
            for key in self._local_keys(state_dict) if key in self.personal_state
        }
        self.model.load_state_dict(overlay, strict=False)
        return self
