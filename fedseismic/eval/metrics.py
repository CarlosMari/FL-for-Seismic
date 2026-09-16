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


def evaluate_loader(model, loader, num_classes, device="cpu"):
    """Return ``(macro_miou, per_class_iou, pixel_acc)`` from loader batches.

    Use this when the loader is a subset of a cube (client local tests). Cube
    reconstruction in ``evaluate`` assumes a full ordered axis-1 scan.
    """
    model.eval()
    model.to(device)
    predictions = []
    targets = []
    with torch.no_grad():
        for images, labels, _ in loader:
            logits = _logits(model(images.to(device, dtype=torch.float)))
            predictions.append(logits.argmax(dim=1).cpu().numpy().ravel())
            targets.append(np.asarray(labels.detach().cpu() if torch.is_tensor(labels) else labels).ravel())
    if not predictions:
        zeros = np.zeros(num_classes, dtype=np.float64)
        return 0.0, zeros, 0.0
    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(targets)
    per_class = jaccard_score(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0,
    )
    pixel_acc = float((y_pred == y_true).mean()) if y_true.size else 0.0
    return float(np.mean(per_class)), np.asarray(per_class), pixel_acc


def score_loader(model, loader, num_classes, device="cpu", task="segmentation"):
    """Primary score is accuracy for classification, macro mIoU for segmentation."""
    miou, per_class, acc = evaluate_loader(model, loader, num_classes, device)
    return (acc if task == "classification" else miou), per_class, acc
