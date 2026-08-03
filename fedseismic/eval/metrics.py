"""The single macro-mIoU implementation used by the new framework."""

import numpy as np
import torch
from sklearn.metrics import jaccard_score


def _logits(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def evaluate(model, loader, labels, device="cpu"):
    """Return ``(macro_miou, per_class_iou)`` for an ordered seismic loader."""
    model.eval()
    model.to(device)
    prediction = np.zeros(labels.shape, dtype=np.int64)
    sample_idx = 0
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device, dtype=torch.float)
            preds = _logits(model(images)).argmax(dim=1)
            for batch_index in range(images.size(0)):
                prediction[:, sample_idx, :] = preds[batch_index].cpu().numpy().T
                sample_idx += 1
    num_classes = int(max(labels.max(initial=0), prediction.max(initial=0)) + 1)
    per_class = jaccard_score(
        labels.flatten(), prediction.flatten(), labels=list(range(num_classes)),
        average=None, zero_division=0,
    )
    return float(per_class.mean()), np.asarray(per_class)
