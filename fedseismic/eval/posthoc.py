"""Test-independent post-hoc logit adjustment and seed ensembling helpers."""

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import jaccard_score


def compute_class_prior(labels, num_classes=None):
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    counts = np.array([(labels == c).sum() for c in range(num_classes)], dtype=np.float64)
    return counts / counts.sum()


def _logits(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _prediction_from_probabilities(probabilities, labels, sample_idx, prediction):
    for batch_index in range(probabilities.size(0)):
        prediction[:, sample_idx + batch_index, :] = probabilities[batch_index].cpu().numpy().T


def evaluate_with_tau(model, loader, labels, log_prior, tau, device=None):
    """Evaluate one model after subtracting ``tau * log_prior`` from logits."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    num_classes = len(log_prior)
    prediction = np.zeros(labels.shape, dtype=np.int64)
    adjust = tau * log_prior.to(device).view(1, num_classes, 1, 1)
    sample_idx = 0
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device, dtype=torch.float)
            preds = (_logits(model(images)) - adjust).argmax(dim=1)
            _prediction_from_probabilities(preds, labels, sample_idx, prediction)
            sample_idx += images.size(0)
    per_class = jaccard_score(
        labels.flatten(), prediction.flatten(), labels=list(range(num_classes)),
        average=None, zero_division=0,
    )
    return float(per_class.mean()), np.asarray(per_class)


def evaluate_ensemble(models, loader, labels, log_prior, tau, device=None):
    """Average adjusted softmax probabilities across models before argmax."""
    if not models:
        raise ValueError("at least one model is required for an ensemble")
    if device is None:
        device = next(models[0].parameters()).device
    for model in models:
        model.eval()
    num_classes = len(log_prior)
    adjust = tau * log_prior.to(device).view(1, num_classes, 1, 1)
    prediction = np.zeros(labels.shape, dtype=np.int64)
    sample_idx = 0
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device, dtype=torch.float)
            probs_sum = None
            for model in models:
                probs = F.softmax(_logits(model(images)) - adjust, dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            preds = (probs_sum / len(models)).argmax(dim=1)
            _prediction_from_probabilities(preds, labels, sample_idx, prediction)
            sample_idx += images.size(0)
    per_class = jaccard_score(
        labels.flatten(), prediction.flatten(), labels=list(range(num_classes)),
        average=None, zero_division=0,
    )
    return float(per_class.mean()), np.asarray(per_class)


def load_checkpoint(model, checkpoint, device="cpu"):
    """Load either a raw state dict or a training checkpoint dictionary."""
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval()


def load_model(checkpoint, model_factory, device="cpu"):
    """Construct and load a model without embedding a project path."""
    return load_checkpoint(model_factory(), checkpoint, device)


def warn_if_test_policy(policy):
    if getattr(policy, "value", policy) == "best_test":
        warnings.warn("BEST_TEST selects on the evaluation set and leaks test data", UserWarning)
