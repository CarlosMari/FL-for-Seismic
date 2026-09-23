"""Honest-but-curious evaluation of personalization vs distribution leakage."""

from .attacks import evaluate_run, flatten_attack_row
from .estimators import last_layer_prior, softmax_prior
from .log import PrivacyLogger, load_labels, load_views
from .metrics import leakage_from_tv, rare_presence_f1, total_variation
from .plot import pareto_mask, plot_boundary

__all__ = [
    "evaluate_run", "flatten_attack_row", "last_layer_prior", "softmax_prior",
    "PrivacyLogger", "load_labels", "load_views", "leakage_from_tv",
    "rare_presence_f1", "total_variation", "pareto_mask", "plot_boundary",
    "expand_jobs", "run_sweep",
]


def expand_jobs(*args, **kwargs):
    from .sweep import expand_jobs as _expand_jobs
    return _expand_jobs(*args, **kwargs)


def run_sweep(*args, **kwargs):
    from .sweep import run_sweep as _run_sweep
    return _run_sweep(*args, **kwargs)
