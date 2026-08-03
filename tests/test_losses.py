import importlib.util
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fedseis = load_script_module("fedseis_losses", "train_fedseis.py")
fedvls = load_script_module("fedvls_losses", "train_fedvls.py")
federated = load_script_module("federated_losses", "train_federated.py")


def test_vacant_class_distillation_one_class_is_positive():
    torch.manual_seed(0)
    local = torch.randn(2, 6, 16, 16)
    global_ = torch.randn(2, 6, 16, 16)
    vacant = torch.zeros(6, dtype=torch.bool)
    vacant[5] = True

    loss = fedseis.vacant_class_distillation(local, global_, vacant)

    assert float(loss) > 0.0


def test_vacant_class_distillation_two_classes_is_positive():
    torch.manual_seed(0)
    local = torch.randn(2, 6, 16, 16)
    global_ = torch.randn(2, 6, 16, 16)
    vacant = torch.zeros(6, dtype=torch.bool)
    vacant[4] = True
    vacant[5] = True

    loss = fedseis.vacant_class_distillation(local, global_, vacant)

    assert float(loss) > 0.0


def test_logit_suppression_large_logits_is_finite():
    torch.manual_seed(0)
    targets = torch.randint(0, 6, (2, 16, 16))
    logits = torch.randn(2, 6, 16, 16) + 200.0
    class_freq = torch.ones(6) / 6

    loss = fedvls.logit_suppression(logits, targets, class_freq)

    assert torch.isfinite(loss)


def test_recall_loss_per_slice_random_input_is_finite_and_non_negative():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 16, 16)
    targets = torch.randint(0, 6, (2, 16, 16))

    loss = federated.RecallLossPerSlice(num_classes=6)(logits, targets)

    assert torch.isfinite(loss)
    assert float(loss) >= 0.0
