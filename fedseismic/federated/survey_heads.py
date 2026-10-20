"""FedPer-survey + FedBN: shared body, per-survey head, local BatchNorm.

Clean Setup B for multi-silo FL with disjoint label spaces. The shared body
is averaged over all sites. The classifier is pooled WITHIN each survey (F3
sites share rows 0-5, Parihaka sites share rows 6-11) so a survey's sites
share head training without cross-survey coupling. BatchNorm stays local.

This is the per-survey-head version of FedPer+BN: instead of each client
keeping its own head (which loses within-survey pooling), the head is shared
at survey granularity. Cross-survey loss coupling is removed by the masked
loss (each site only trains its own rows).
"""

from copy import deepcopy

import numpy as np
import torch

from .client import _dataset_labels, _is_bn_key, _logits
from .fedper import FedPerClientTrainer, personalization_keys


def survey_rows_from_info(client_info, num_classes):
    """Row indices each site trains: classes with positive mass in its mix."""
    fracs = np.asarray(client_info["class_fracs"], dtype=np.float64).reshape(-1)
    return np.flatnonzero(fracs > 0).tolist()


class FedPerSurveyBNClientTrainer(FedPerClientTrainer):
    """Per-survey head + local BN: masked loss, trained head uploaded.

    Like FedPer+BN but the head is shared at survey granularity: the site
    uploads its trained rows (not the global head) so ``aggregate_survey_heads``
    can pool each row within its survey. Cross-survey loss coupling is removed
    by masking the loss to the site's own classes. BN stays local (the upload
    carries a global BN placeholder that the aggregator ignores).
    """

    def _local_keys(self, state_dict):
        return {key for key in state_dict if _is_bn_key(key)}

    def _own_classes(self):
        labels = _dataset_labels(getattr(self.loader, "dataset", None))
        if labels is None:
            return None
        present = np.unique(np.asarray(labels).reshape(-1).astype(int))
        return torch.as_tensor(present, dtype=torch.long)

    def train(self, global_state=None):
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
        trained = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        # The deployed per-survey model predicts only its own classes: push
        # the untrained rows down so an unmasked argmax can't pick them.
        if own is not None:
            bias_key = "outc.conv.bias"
            weight_key = "outc.conv.weight"
            if bias_key in trained and weight_key in trained:
                keep = torch.zeros(trained[bias_key].shape[0], dtype=torch.bool)
                keep[own] = True
                trained[bias_key] = torch.where(
                    keep, trained[bias_key],
                    torch.as_tensor(-1e9, dtype=trained[bias_key].dtype),
                )
                weight_shape = (-1,) + (1,) * (trained[weight_key].ndim - 1)
                trained[weight_key] = torch.where(
                    keep.view(weight_shape), trained[weight_key],
                    torch.zeros_like(trained[weight_key]),
                )
        self.personal_state = trained
        # Upload: body + trained head (own rows) + global BN placeholder.
        uploaded = deepcopy(trained)
        for key in self._local_keys(trained):
            uploaded[key] = global_state[key].detach().cpu().clone()
        self._upload_state = uploaded
        self.model.load_state_dict(uploaded)
        return self

    def download(self, state_dict):
        # Load the shared body + the per-survey pooled head from the aggregate,
        # then restore ONLY local BN. The head must come from the aggregate
        # (shared within survey), not from personal_state — otherwise the
        # per-survey pooling is thrown away each round and the head is per-client.
        super(FedPerClientTrainer, self).download(state_dict)
        if self.personal_state is None:
            return self
        overlay = {
            key: self.personal_state[key]
            for key in self._local_keys(state_dict) if key in self.personal_state
        }
        self.model.load_state_dict(overlay, strict=False)
        return self


def aggregate_survey_heads(states, client_infos, num_classes):
    """Body averaged over all sites; outc rows averaged within each survey.

    ``states`` are the uploaded state dicts (body + trained head rows + a
    global BN placeholder). ``client_infos[i]["class_fracs"]`` gives site i's
    class support. Row k is averaged only over sites that train row k (its
    survey's sites); other sites' row k is the global placeholder and is
    excluded so it cannot dilute the trained value. BatchNorm keys are left
    at the first state's value (each site overrides its own on download).
    """
    if not states:
        raise ValueError("no states to aggregate")
    out = deepcopy(states[0])
    for key in out:
        if _is_bn_key(key):
            continue  # local normalisation: leave the placeholder
        tensors = [np.asarray(state[key], dtype=np.float64) for state in states]
        stacked = np.stack(tensors)
        if key.startswith("outc.") and np.asarray(out[key]).ndim >= 1:
            # Per-row mean over the sites that train that row.
            result = stacked[0].copy()
            is_weight = np.asarray(out[key]).ndim == 2  # (C, D); bias is (C,)
            for row in range(min(num_classes, stacked.shape[1] if stacked.ndim > 1 else 0)):
                members = [
                    i for i, info in enumerate(client_infos)
                    if row in survey_rows_from_info(info, num_classes)
                ]
                if not members:
                    continue
                if is_weight:  # weight (C, D)
                    result[row] = stacked[members, row].mean(axis=0)
                else:  # bias (C,)
                    result[row] = stacked[members, row].mean()
            out[key] = torch.as_tensor(result, dtype=out[key].dtype)
        else:
            out[key] = torch.as_tensor(np.mean(stacked, axis=0), dtype=out[key].dtype)
    return out
