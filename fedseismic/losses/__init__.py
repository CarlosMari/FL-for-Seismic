"""Loss registry for the segmentation training loop."""

from .dice import DiceLoss, FocalDiceLoss
from .fciou import FCIoULoss
from .focal import FocalLoss
from .recall import (
    CompoundCrosslineRecallLoss,
    CrosslineRecallLoss,
    RecallLoss,
    RecallLossPerSlice,
)
from .tversky import AsymmetricTverskyLoss, UnifiedFocalLoss


LOSSES = {
    "focal": FocalLoss,
    "dice": DiceLoss,
    "focaldice": FocalDiceLoss,
    "tversky": AsymmetricTverskyLoss,
    "unifiedfocal": UnifiedFocalLoss,
    "crossrecall": CompoundCrosslineRecallLoss,
    "recall": RecallLoss,
    "recall_slice": RecallLossPerSlice,
    "fciou": FCIoULoss,
}

__all__ = [
    "LOSSES", "FocalLoss", "DiceLoss", "FocalDiceLoss", "AsymmetricTverskyLoss",
    "UnifiedFocalLoss", "CrosslineRecallLoss", "CompoundCrosslineRecallLoss",
    "RecallLoss", "RecallLossPerSlice", "FCIoULoss",
]
