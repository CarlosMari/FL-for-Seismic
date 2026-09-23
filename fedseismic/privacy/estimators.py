"""Histogram estimators that an honest-but-curious server can run on uploads."""

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge

from fedseismic.eval.metrics import _logits
from fedseismic.federated.aggregation import is_batchnorm_key

from .metrics import to_simplex


def softmax_prior(model, loader, num_classes, device="cpu", max_batches=None, start_batch=0):
    """Mean predicted class prior on an unlabeled public probe set.

    ``start_batch`` skips the batches FedKPer's aggregation weight already saw.
    ``max_batches`` counts only the batches after that skip.
    """
    if loader is None:
        raise ValueError("softmax_prior needs a probe loader")
    model.eval()
    model.to(device)
    total = torch.zeros(num_classes, device=device, dtype=torch.float64)
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index < start_batch:
                continue
            if max_batches is not None and count >= max_batches:
                break
            images = batch[0].to(device, dtype=torch.float)
            logits = _logits(model(images))
            probs = F.softmax(logits, dim=1)
            if probs.ndim == 4:
                total = total + probs.mean(dim=(0, 2, 3)).double()
            else:
                total = total + probs.mean(dim=0).double()
            count += 1
    if count == 0:
        return np.full(num_classes, 1.0 / max(num_classes, 1), dtype=np.float64)
    return (total / count).detach().cpu().numpy()


def mean_logits(model, loader, num_classes, device="cpu", max_batches=None, start_batch=0):
    """Mean logits on the same probe window as ``softmax_prior``."""
    if loader is None:
        raise ValueError("mean_logits needs a probe loader")
    model.eval()
    model.to(device)
    total = torch.zeros(num_classes, device=device, dtype=torch.float64)
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index < start_batch:
                continue
            if max_batches is not None and count >= max_batches:
                break
            images = batch[0].to(device, dtype=torch.float)
            logits = _logits(model(images))
            if logits.ndim == 4:
                total = total + logits.mean(dim=(0, 2, 3)).double()
            else:
                total = total + logits.mean(dim=0).double()
            count += 1
    if count == 0:
        return np.zeros(num_classes, dtype=np.float64)
    return (total / count).detach().cpu().numpy()


def extract_last_linear(state):
    """Return ``(weight, bias)`` for the last class-scoring tensor in ``state``."""
    items = list(state.items())
    linear = [
        (key, tensor) for key, tensor in items
        if key.endswith("weight") and torch.as_tensor(tensor).ndim == 2
    ]
    if linear:
        key, weight = linear[-1]
        bias_key = key[: -len("weight")] + "bias"
        bias = state.get(bias_key)
        weight_np = torch.as_tensor(weight).detach().cpu().float().numpy()
        bias_np = (
            torch.as_tensor(bias).detach().cpu().float().numpy()
            if bias is not None else None
        )
        return weight_np, bias_np
    conv = [
        (key, tensor) for key, tensor in items
        if key.endswith("weight") and torch.as_tensor(tensor).ndim == 4
    ]
    if not conv:
        return None, None
    key, weight = conv[-1]
    bias_key = key[: -len("weight")] + "bias"
    bias = state.get(bias_key)
    weight_np = torch.as_tensor(weight).detach().cpu().float().numpy()
    bias_np = (
        torch.as_tensor(bias).detach().cpu().float().numpy()
        if bias is not None else None
    )
    return weight_np, bias_np


def last_layer_prior(state, num_classes=None):
    """Softmax of per-class last-layer energy (row L2 plus bias)."""
    weight, bias = extract_last_linear(state)
    if weight is None:
        if num_classes is None:
            raise ValueError("no linear/conv classifier tensor in state")
        return np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
    if weight.ndim == 2:
        scores = np.linalg.norm(weight, axis=1)
    else:
        scores = np.linalg.norm(weight.reshape(weight.shape[0], -1), axis=1)
    if bias is not None:
        scores = scores + bias.reshape(-1)
    if num_classes is not None and len(scores) != num_classes:
        scores = scores[:num_classes] if len(scores) > num_classes else np.pad(
            scores, (0, num_classes - len(scores))
        )
    shifted = scores - np.max(scores)
    return to_simplex(np.exp(shifted))


def flatten_state_delta(local_state, global_state):
    parts = []
    for key, tensor in local_state.items():
        if key not in global_state:
            continue
        delta = (
            torch.as_tensor(tensor).detach().cpu().float()
            - torch.as_tensor(global_state[key]).detach().cpu().float()
        )
        parts.append(delta.numpy().ravel())
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)


def project_vector(vector, dim, seed):
    """Fixed-basis count sketch so every client in a run shares one projection."""
    vector = np.asarray(vector, dtype=np.float32).ravel()
    if dim is None or dim >= vector.size:
        return vector
    out = np.zeros(dim, dtype=np.float64)
    rng = np.random.RandomState(int(seed))
    offset = 0
    chunk = 1_000_000
    while offset < vector.size:
        end = min(offset + chunk, vector.size)
        length = end - offset
        buckets = rng.randint(0, dim, size=length)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=length)
        np.add.at(out, buckets, vector[offset:end].astype(np.float64) * signs)
        offset = end
    return out.astype(np.float32)


def batchnorm_tensors(state):
    return {
        key: torch.as_tensor(value).detach().cpu().float().numpy()
        for key, value in state.items()
        if is_batchnorm_key(key)
    }


def example_membership_scores(model, loader, device="cpu", max_examples=None):
    """Per-example ``(-loss, true-class confidence)`` used as MIA scores."""
    model.eval()
    model.to(device)
    losses = []
    confidences = []
    seen = 0
    with torch.no_grad():
        for images, targets, *_ in loader:
            if max_examples is not None and seen >= max_examples:
                break
            images = images.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)
            logits = _logits(model(images))
            if logits.ndim == 4:
                per_pixel = F.cross_entropy(logits, targets, reduction="none")
                loss = per_pixel.view(per_pixel.size(0), -1).mean(dim=1)
                probs = F.softmax(logits, dim=1)
                conf = probs.max(dim=1).values.view(probs.size(0), -1).mean(dim=1)
            else:
                loss = F.cross_entropy(logits, targets, reduction="none")
                probs = F.softmax(logits, dim=1)
                conf = probs.gather(1, targets.view(-1, 1)).squeeze(1)
            losses.append(loss.detach().cpu())
            confidences.append(conf.detach().cpu())
            seen += images.size(0)
    if not losses:
        return np.zeros(0), np.zeros(0)
    return torch.cat(losses).numpy(), torch.cat(confidences).numpy()


class HistogramRidge:
    """Leave-one-client-out regressor from a projected update to π."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def predict_loo(self, features, histograms):
        features = np.asarray(features, dtype=np.float64)
        histograms = np.asarray(histograms, dtype=np.float64)
        if len(features) != len(histograms):
            raise ValueError("features and histograms must align")
        if len(features) < 2:
            return [to_simplex(hist) for hist in histograms]
        preds = []
        for holdout in range(len(features)):
            train_idx = [index for index in range(len(features)) if index != holdout]
            model = Ridge(alpha=self.alpha)
            model.fit(features[train_idx], histograms[train_idx])
            raw = model.predict(features[holdout:holdout + 1])[0]
            preds.append(to_simplex(raw))
        return preds


def clone_model_from_state(factory, state, device="cpu"):
    model = factory()
    model.load_state_dict(deepcopy(state))
    model.to(device)
    model.eval()
    return model
