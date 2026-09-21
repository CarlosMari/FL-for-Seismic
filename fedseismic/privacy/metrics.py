"""Distribution-inference and leakage scores used by the privacy pipeline."""

import numpy as np
from sklearn.metrics import f1_score


def to_simplex(values, eps=1e-12):
    histogram = np.asarray(values, dtype=np.float64).reshape(-1)
    histogram = np.clip(histogram, 0.0, None)
    total = float(histogram.sum())
    if total <= eps or not np.isfinite(total):
        return np.ones_like(histogram) / max(len(histogram), 1)
    return histogram / total


def total_variation(pi_true, pi_hat):
    left = to_simplex(pi_true)
    right = to_simplex(pi_hat)
    if left.shape != right.shape:
        raise ValueError("histograms must have the same number of classes")
    return 0.5 * float(np.abs(left - right).sum())


def kl_divergence(pi_true, pi_hat, eps=1e-12):
    left = to_simplex(pi_true)
    right = to_simplex(pi_hat)
    return float(np.sum(left * np.log((left + eps) / (right + eps))))


def leakage_from_tv(tv):
    """Higher means the histogram was easier to infer (L = 1 - TV)."""
    return 1.0 - float(tv)


def presence_mask(pi, threshold=1e-3):
    return (to_simplex(pi) > threshold).astype(np.int32)


def presence_f1(pi_true, pi_hat, threshold=1e-3):
    true_mask = presence_mask(pi_true, threshold=threshold)
    pred_mask = presence_mask(pi_hat, threshold=threshold)
    if true_mask.size == 0:
        return 0.0
    return float(f1_score(true_mask, pred_mask, zero_division=0))


def rare_presence_f1(pi_true, pi_hat, rare_classes, threshold=1e-3):
    """F1 over the rare-class subset (seismic facies 4/5 by default)."""
    true_hist = to_simplex(pi_true)
    pred_hist = to_simplex(pi_hat)
    rare = np.asarray(
        [index for index in rare_classes if 0 <= int(index) < len(true_hist)],
        dtype=np.int64,
    )
    if rare.size == 0:
        return 0.0
    true_mask = (true_hist[rare] > threshold).astype(np.int32)
    pred_mask = (pred_hist[rare] > threshold).astype(np.int32)
    if true_mask.size == 1:
        return float(true_mask[0] == pred_mask[0])
    if true_mask.sum() == 0 and pred_mask.sum() == 0:
        return 1.0
    return float(f1_score(true_mask, pred_mask, zero_division=0))


def mean_metric(pairs, metric):
    if not pairs:
        return float("nan")
    return float(np.mean([metric(true, hat) for true, hat in pairs]))
