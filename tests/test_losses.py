import torch
import torch.nn.functional as F

from fedseismic.losses import (
    AsymmetricTverskyLoss,
    CompoundCrosslineRecallLoss,
    CrosslineRecallLoss,
    DiceLoss,
    FCIoULoss,
    FocalDiceLoss,
    FocalLoss,
    RecallLoss,
    RecallLossPerSlice,
    UnifiedFocalLoss,
)


def test_focal_loss_matches_reference_formula():
    torch.manual_seed(42)
    logits = torch.randn(2, 6, 4, 5)
    targets = torch.randint(0, 6, (2, 4, 5))
    ce = F.cross_entropy(logits, targets, reduction="none")
    expected = (((1 - torch.exp(-ce)) ** 2) * ce).mean()
    assert torch.equal(FocalLoss()(logits, targets), expected)


def test_all_losses_have_finite_outputs():
    torch.manual_seed(7)
    logits = torch.randn(2, 6, 4, 5)
    targets = torch.randint(0, 6, (2, 4, 5))
    losses = [
        FocalLoss(), DiceLoss(), FocalDiceLoss(),
        AsymmetricTverskyLoss([.5] * 6, [.5] * 6),
        UnifiedFocalLoss([.5] * 6, [.5] * 6),
        CrosslineRecallLoss([4, 5]), CompoundCrosslineRecallLoss([4, 5]),
        RecallLoss(6), RecallLossPerSlice(6), FCIoULoss(),
    ]
    for loss in losses:
        value = loss(logits, targets)
        assert value.ndim == 0
        assert torch.isfinite(value)


def test_vacant_slice_and_large_logits_edges():
    from fedseismic.federated.client import logit_suppression, vacant_class_distillation

    local = torch.tensor([[[[2.0]], [[0.0]], [[-1.0]]]])
    global_logits = torch.tensor([[[[0.0]], [[2.0]], [[-1.0]]]])
    assert vacant_class_distillation(local, global_logits, torch.tensor([False, True, False])) > 0
    logits = torch.full((1, 3, 2, 2), 200.0)
    assert torch.isfinite(logit_suppression(logits, torch.zeros((1, 2, 2), dtype=torch.long),
                                            torch.ones(3)))
