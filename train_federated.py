"""
Federated Averaging (FedAvg) Simulation on Parihaka Seismic Dataset
===================================================================
Supports both IID (random shuffle) and Non-IID (geographic/spatial)
data partitioning across a configurable number of clients.

Usage:
    # Non-IID, 3 clients (default — what we already ran)
    python train_federated.py --num_clients 3 --split noniid

    # IID, 3 clients
    python train_federated.py --num_clients 3 --split iid

    # Non-IID, 5 clients
    python train_federated.py --num_clients 5 --split noniid

    # IID, 5 clients
    python train_federated.py --num_clients 5 --split iid
"""

import argparse
import os
import sys
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.transforms import transforms
from sklearn.metrics import jaccard_score
import wandb

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

PARIHAKA_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "OCT_SEISMIC_PIPELINE"))
TRAIN_SEISMIC = os.path.join(PROJECT_ROOT, "datasets", "train", "train_seismic.npy")
TRAIN_LABELS  = os.path.join(PROJECT_ROOT, "datasets", "train", "train_labels.npy")
TEST1_SEISMIC = os.path.join(PROJECT_ROOT, "datasets", "test_once", "test1_seismic.npy")
TEST1_LABELS  = os.path.join(PROJECT_ROOT, "datasets", "test_once", "test1_labels.npy")
TEST2_SEISMIC = os.path.join(PROJECT_ROOT, "datasets", "test_once", "test2_seismic.npy")
TEST2_LABELS  = os.path.join(PROJECT_ROOT, "datasets", "test_once", "test2_labels.npy")

from train_tools.models.unet import UNet
from train_tools.preprocessing.seismic.datasets import InlineLoader

# ── Fixed hyperparameters ───────────────────────────────────────────────────
NUM_ROUNDS    = 20
LOCAL_EPOCHS  = 3
BATCH_SIZE    = 4
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
NUM_CLASSES   = 6
DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED          = 42


# ── Loss functions ──────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal = alpha_t * focal
        return focal.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=probs.shape[1])
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=1.0, dice_weight=1.0, gamma=2.0, alpha=None):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.fw = focal_weight
        self.dw = dice_weight

    def forward(self, logits, targets):
        return self.fw * self.focal(logits, targets) + self.dw * self.dice(logits, targets)


class AsymmetricTverskyLoss(nn.Module):
    """
    Tversky loss with asymmetric alpha/beta per class.
    alpha weights FP, beta weights FN. For rare classes we set beta >> alpha
    to heavily penalize missed (false-negative) pixels.
      L_c = 1 - (TP_c + s) / (TP_c + alpha_c * FP_c + beta_c * FN_c + s)
    Optional focal exponent gamma: L = sum( L_c ** (1/gamma) ).
    """
    def __init__(self, alpha_per_class, beta_per_class, gamma=1.0, smooth=1.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha_per_class, dtype=torch.float))
        self.register_buffer("beta", torch.tensor(beta_per_class, dtype=torch.float))
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        tp = (probs * one_hot).sum(dim=dims)
        fp = (probs * (1 - one_hot)).sum(dim=dims)
        fn = ((1 - probs) * one_hot).sum(dim=dims)
        alpha = self.alpha.to(logits.device)
        beta = self.beta.to(logits.device)
        t_idx = (tp + self.smooth) / (tp + alpha * fp + beta * fn + self.smooth)
        loss = (1.0 - t_idx) ** (1.0 / self.gamma) if self.gamma != 1.0 else (1.0 - t_idx)
        return loss.mean()


class UnifiedFocalLoss(nn.Module):
    """
    Yeung et al.'s Unified Focal Loss: combine symmetric focal CE (down-weight
    easy examples) with asymmetric focal Tversky (down-weight well-segmented
    majority classes, emphasize minority recall).
      L = lambda_ce * focal_ce + (1 - lambda_ce) * focal_tversky
    """
    def __init__(self, alpha_per_class, beta_per_class, gamma_ce=2.0,
                 gamma_t=2.0, lambda_ce=0.5, smooth=1.0):
        super().__init__()
        self.focal_ce = FocalLoss(gamma=gamma_ce)
        self.focal_tv = AsymmetricTverskyLoss(
            alpha_per_class=alpha_per_class, beta_per_class=beta_per_class,
            gamma=gamma_t, smooth=smooth,
        )
        self.lambda_ce = lambda_ce

    def forward(self, logits, targets):
        return (self.lambda_ce * self.focal_ce(logits, targets)
                + (1.0 - self.lambda_ce) * self.focal_tv(logits, targets))


class CrosslineRecallLoss(nn.Module):
    """
    Novel seismic-specific auxiliary loss.
    For each slice in a batch whose ground truth contains a rare class, compute
    per-slice recall for that class and add (1 - recall). Missing a rare class
    across an entire slice is penalized as 1.0 (full recall miss), regardless
    of how small that class's footprint was on that slice.
      L_crl = mean over slices containing any rare class of mean over those classes of (1 - recall)
    Always combined with a base segmentation loss (see CompoundCrosslineRecallLoss).
    """
    def __init__(self, rare_classes, smooth=1.0):
        super().__init__()
        self.rare_classes = rare_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        B = targets.shape[0]
        penalties = []
        for b in range(B):
            tgt_b = targets[b]
            prd_b = preds[b]
            prob_b = probs[b]
            slice_terms = []
            for c in self.rare_classes:
                gt_mask = (tgt_b == c)
                if gt_mask.sum() == 0:
                    continue  # class not in this slice — nothing to penalize
                # Soft recall via predicted probability on GT pixels (differentiable)
                recall = (prob_b[c] * gt_mask.float()).sum() / (gt_mask.float().sum() + self.smooth)
                slice_terms.append(1.0 - recall)
            if slice_terms:
                penalties.append(torch.stack(slice_terms).mean())
        if not penalties:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return torch.stack(penalties).mean()


class CompoundCrosslineRecallLoss(nn.Module):
    """Base (Focal+Dice) loss + lambda * CrosslineRecallLoss."""
    def __init__(self, rare_classes, lambda_crl=1.0, gamma=2.0, alpha=None):
        super().__init__()
        self.base = FocalDiceLoss(gamma=gamma, alpha=alpha)
        self.crl = CrosslineRecallLoss(rare_classes=rare_classes)
        self.lambda_crl = lambda_crl

    def forward(self, logits, targets):
        return self.base(logits, targets) + self.lambda_crl * self.crl(logits, targets)


class RecallLoss(nn.Module):
    """
    Tian et al. "Recall Loss for Imbalanced Image Classification and
    Semantic Segmentation" (OpenReview SlprFTIQP3).
    Standard cross-entropy reweighted per-class by instantaneous
    false-negative rate: w_c = FN_c / (FN_c + TP_c) = 1 - recall_c.
    Classes with poor recall get upweighted; classes already at high
    recall approach zero weight (gradient flows to where it's needed).
    """
    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        preds = logits.argmax(dim=1)
        w = torch.zeros(self.num_classes, device=logits.device)
        for c in range(self.num_classes):
            gt_c = (targets == c)
            n_gt = gt_c.sum().float()
            if n_gt < 1:
                w[c] = 0.0
                continue
            tp = ((preds == c) & gt_c).sum().float()
            fn = n_gt - tp
            w[c] = fn / (fn + tp + self.smooth)
        ce = F.cross_entropy(logits, targets, weight=w.detach(), reduction="mean")
        return ce


class RecallLossPerSlice(nn.Module):
    """
    Per-slice variant of Tian et al. Recall Loss. For each 2D crossline slice
    in the batch, compute per-class FNR weights from that slice's TPs/FNs, then
    compute CE on the slice with those weights. Final loss is mean over slices.
    Difference from vanilla RecallLoss (batch-level FNR): locality. A slice
    where a rare class is present but being missed will get a stronger per-class
    weight than the batch average would suggest. If slice-level wins over
    batch-level, locality of recall computation is the contribution; if it
    doesn't, CRL's novelty claim evaporates.
    """
    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        B = targets.shape[0]
        preds = logits.argmax(dim=1)
        losses = []
        for b in range(B):
            tgt_b = targets[b]
            prd_b = preds[b]
            w = torch.zeros(self.num_classes, device=logits.device)
            for c in range(self.num_classes):
                gt_c = (tgt_b == c)
                n_gt = gt_c.sum().float()
                if n_gt < 1:
                    w[c] = 0.0
                    continue
                tp = ((prd_b == c) & gt_c).sum().float()
                fn = n_gt - tp
                w[c] = fn / (fn + tp + self.smooth)
            ce_b = F.cross_entropy(
                logits[b:b+1], targets[b:b+1], weight=w.detach(), reduction="mean",
            )
            losses.append(ce_b)
        return torch.stack(losses).mean()


class FCIoULoss(nn.Module):
    """
    Sheffield et al. "FCIoU: A Targeted Approach for Improving Minority
    Class Detection in Semantic Segmentation Systems" (MAKE 2023).
    Focal class-based IoU: per-class soft-IoU with focal reweighting
    so well-segmented (majority) classes contribute less gradient.
      L = mean_c (1 - IoU_c) ** gamma * (1 - IoU_c)
    """
    def __init__(self, gamma=2.0, smooth=1.0):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        union = probs.sum(dim=dims) + one_hot.sum(dim=dims) - intersection
        iou_per_class = (intersection + self.smooth) / (union + self.smooth)
        loss_per_class = (1.0 - iou_per_class) ** self.gamma * (1.0 - iou_per_class)
        return loss_per_class.mean()


# ── Data partitioning ──────────────────────────────────────────────────────
def partition_noniid(num_crosslines, num_clients):
    """Geographic/spatial split: contiguous chunks of crosslines per client."""
    chunk_size = num_crosslines // num_clients
    partitions = []
    for c in range(num_clients):
        start = c * chunk_size
        end = (c + 1) * chunk_size if c < num_clients - 1 else num_crosslines
        partitions.append(list(range(start, end)))
    return partitions


def partition_iid(num_crosslines, num_clients, rng):
    """IID split: randomly shuffle crosslines and deal them out equally."""
    all_idxs = np.arange(num_crosslines)
    rng.shuffle(all_idxs)
    partitions = []
    chunk_size = num_crosslines // num_clients
    for c in range(num_clients):
        start = c * chunk_size
        end = (c + 1) * chunk_size if c < num_clients - 1 else num_crosslines
        # Sort within each client so DataLoader gets proper crossline indices
        partitions.append(sorted(all_idxs[start:end].tolist()))
    return partitions


# ── Data loading ────────────────────────────────────────────────────────────
def load_and_normalize(path):
    arr = np.load(path)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def build_client_loaders(train_seismic, train_labels, partitions, batch_size):
    """Build DataLoaders from a list of crossline index lists."""
    transform = transforms.Compose([transforms.ToTensor()])
    client_loaders = []
    for c, crossline_idxs in enumerate(partitions):
        dataset = InlineLoader(
            seismic_cube=train_seismic,
            label_cube=train_labels,
            inline_inds=crossline_idxs,
            train_status=True,
            transform=transform,
        )
        loader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        client_loaders.append(loader)
        if len(crossline_idxs) <= 10:
            idx_str = str(crossline_idxs)
        else:
            idx_str = f"[{crossline_idxs[0]}..{crossline_idxs[-1]}]"
        print(f"  Client {c}: {len(crossline_idxs)} crosslines {idx_str}")
    return client_loaders


def build_test_loader(seismic_path, labels_path, batch_size=1):
    seismic = load_and_normalize(seismic_path)
    labels = np.load(labels_path)
    num_crosslines = seismic.shape[1]
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = InlineLoader(
        seismic_cube=seismic, label_cube=labels,
        inline_inds=list(range(num_crosslines)),
        train_status=False, transform=transform,
    )
    loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    return loader, labels


# ── Evaluation ──────────────────────────────────────────────────────────────
def evaluate_global(model, test_loader, test_labels, device):
    model.eval()
    model.to(device)
    prediction = np.zeros(test_labels.shape)
    sample_idx = 0
    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device, dtype=torch.float)
            logits, _ = model(images)
            preds = logits.argmax(dim=1)
            for b in range(images.size(0)):
                prediction[:, sample_idx, :] = preds[b].cpu().numpy().T
                sample_idx += 1
    per_class_iou = jaccard_score(
        test_labels.flatten(), prediction.flatten(),
        labels=list(range(NUM_CLASSES)), average=None
    )
    miou = per_class_iou.mean()
    return miou, per_class_iou


# ── FedAvg helpers ──────────────────────────────────────────────────────────
def fedavg_aggregate(state_dicts):
    n = len(state_dicts)
    avg = copy.deepcopy(state_dicts[0])
    for key in avg.keys():
        for i in range(1, n):
            avg[key] = avg[key] + state_dicts[i][key]
        avg[key] = avg[key] / n
    return avg


def weighted_aggregate(state_dicts, weights):
    """Weighted average of state dicts. Weights are normalized to sum to 1."""
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    avg = copy.deepcopy(state_dicts[0])
    for key in avg.keys():
        avg[key] = avg[key] * w[0]
        for i in range(1, len(state_dicts)):
            avg[key] = avg[key] + state_dicts[i][key] * w[i]
    return avg


RARE_CLASSES = [4, 5]  # Classes 4 and 5 are rare in Parihaka
# Global train-set class prior (computed once from train_labels.npy, Parihaka)
TRAIN_CLASS_PRIOR = np.array([0.28093788, 0.1188557, 0.48592013, 0.0664164,
                              0.03278635, 0.01508355], dtype=np.float64)
RARE_FRAC_THRESHOLD = 1e-3  # skip classwise mIoU contribution if client has < 0.1% pixels of class


def compute_client_class_info(train_labels, partitions):
    """Precompute per-client class counts for diversity/rare weighting."""
    client_info = []
    for c, idxs in enumerate(partitions):
        client_labels = train_labels[:, idxs, :].flatten()
        unique_classes = set(np.unique(client_labels).tolist())
        class_counts = dict(zip(*np.unique(client_labels, return_counts=True)))
        total = client_labels.size
        rare_fraction = sum(class_counts.get(rc, 0) for rc in RARE_CLASSES) / total
        class_fracs = np.zeros(NUM_CLASSES, dtype=np.float64)
        for c_idx in range(NUM_CLASSES):
            class_fracs[c_idx] = class_counts.get(c_idx, 0) / max(total, 1)
        client_info.append({
            "num_classes": len(unique_classes),
            "has_classes": unique_classes,
            "rare_fraction": rare_fraction,
            "class_fracs": class_fracs,
        })
    return client_info


def get_agg_weights(strategy, selected_clients, client_info,
                    client_models=None, test_loader=None, test_labels=None, device=None,
                    client_class_ious=None):
    """Compute aggregation weights for selected clients based on strategy.

    client_class_ious: optional list of np.array[NUM_CLASSES] with per-class IoU on
                       each client's local training data, in the same order as
                       selected_clients. Required for rare_miou / invfreq_miou.
    """
    eps = 1e-3
    if strategy == "equal":
        return [1.0] * len(selected_clients)

    elif strategy == "diversity":
        # Weight by number of unique classes
        weights = [client_info[c]["num_classes"] for c in selected_clients]
        return weights

    elif strategy == "rare":
        # Weight by rare-class fraction (with floor so clients without rare classes still contribute)
        fracs = [client_info[c]["rare_fraction"] for c in selected_clients]
        # Floor: every client gets at least 0.1 base weight
        weights = [max(f, 0.01) for f in fracs]
        return weights

    elif strategy == "rare_miou":
        # V1: w_k = (Σ_{c∈R} f_c^k + ε) × mean_{c∈R, gated} m_c^k
        assert client_class_ious is not None, "rare_miou needs client_class_ious"
        weights = []
        for i, c in enumerate(selected_clients):
            fracs = client_info[c]["class_fracs"]
            rare_frac_sum = sum(fracs[rc] for rc in RARE_CLASSES)
            ious = client_class_ious[i]
            gated = [ious[rc] for rc in RARE_CLASSES if fracs[rc] > RARE_FRAC_THRESHOLD]
            miou_rare = float(np.mean(gated)) if len(gated) > 0 else 0.0
            w = (rare_frac_sum + eps) * (miou_rare + eps)
            weights.append(w)
        return weights

    elif strategy == "invfreq_miou":
        # V3: w_k = Σ_{c∈R} (f_c^k / π_c) × m_c^k  + ε
        assert client_class_ious is not None, "invfreq_miou needs client_class_ious"
        weights = []
        for i, c in enumerate(selected_clients):
            fracs = client_info[c]["class_fracs"]
            ious = client_class_ious[i]
            w = 0.0
            for rc in RARE_CLASSES:
                if fracs[rc] > RARE_FRAC_THRESHOLD:
                    w += (fracs[rc] / max(TRAIN_CLASS_PRIOR[rc], 1e-8)) * ious[rc]
            weights.append(w + eps)
        return weights

    elif strategy == "invfreq":
        # V2: w_k = Σ_{c∈R} (f_c^k / π_c)  + ε  (inverse-freq, no mIoU signal)
        weights = []
        for c in selected_clients:
            fracs = client_info[c]["class_fracs"]
            w = sum(fracs[rc] / max(TRAIN_CLASS_PRIOR[rc], 1e-8) for rc in RARE_CLASSES)
            weights.append(w + eps)
        return weights

    elif strategy == "invfreq_invmiou":
        # Boost-bad variant: w_k = Σ_{c∈R} (f_c^k / π_c) × (1 - m_c^k) + ε
        # Pulls aggregate toward clients that have rare pixels AND are struggling
        # on them (coverage/deficit theory vs V3's signal-dilution theory).
        assert client_class_ious is not None, "invfreq_invmiou needs client_class_ious"
        weights = []
        for i, c in enumerate(selected_clients):
            fracs = client_info[c]["class_fracs"]
            ious = client_class_ious[i]
            w = 0.0
            for rc in RARE_CLASSES:
                if fracs[rc] > RARE_FRAC_THRESHOLD:
                    w += (fracs[rc] / max(TRAIN_CLASS_PRIOR[rc], 1e-8)) * (1.0 - ious[rc])
            weights.append(w + eps)
        return weights

    elif strategy == "accuracy":
        # Weight by mIoU of each client's local model on test set
        assert client_models is not None and test_loader is not None
        weights = []
        for model in client_models:
            miou, _ = evaluate_global(model, test_loader, test_labels, device)
            weights.append(max(miou, 0.01))  # floor to avoid zero weight
        return weights

    else:
        return [1.0] * len(selected_clients)


def local_train(model, loader, criterion, device, local_epochs, lr, weight_decay):
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(local_epochs):
        for images, targets, _ in loader:
            images = images.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)
            logits, _ = model(images)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return copy.deepcopy(model.state_dict())


def compute_local_class_iou(model, loader, device):
    """Compute per-class IoU on a client's local training data (after local training).
    Returns np.array[NUM_CLASSES] with IoU in [0, 1]; classes absent from local data
    show up as 0 via jaccard_score's labels= argument (with zero_division=0).
    """
    model.eval()
    model.to(device)
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device, dtype=torch.float)
            logits, _ = model(images)
            preds = logits.argmax(dim=1).cpu().numpy().flatten()
            all_preds.append(preds)
            all_targets.append(targets.numpy().flatten())
    if not all_preds:
        return np.zeros(NUM_CLASSES, dtype=np.float64)
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    per_class = jaccard_score(
        targets, preds,
        labels=list(range(NUM_CLASSES)), average=None, zero_division=0,
    )
    return np.asarray(per_class, dtype=np.float64)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FedAvg on Parihaka")
    parser.add_argument("--num_clients", type=int, default=3, help="Number of FL clients (default: 3)")
    parser.add_argument("--split", type=str, default="noniid", choices=["iid", "noniid"],
                        help="Data partition: 'iid' (random) or 'noniid' (geographic)")
    parser.add_argument("--num_rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--local_epochs", type=int, default=LOCAL_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sample_ratio", type=float, default=1.0,
                        help="Fraction of clients sampled per round (default: 1.0 = all)")
    parser.add_argument("--class_weights", action="store_true",
                        help="Use global inverse-frequency class weights in FocalLoss")
    parser.add_argument("--agg_strategy", type=str, default="equal",
                        choices=["equal", "accuracy", "diversity", "rare",
                                 "rare_miou", "invfreq_miou", "invfreq",
                                 "invfreq_invmiou"],
                        help="Aggregation weighting: equal (FedAvg), accuracy (mIoU), "
                             "diversity (#classes), rare (rare-class pixel fraction), "
                             "rare_miou (rare-fraction × classwise local mIoU), "
                             "invfreq_miou (inverse-freq rare-class × classwise mIoU), "
                             "invfreq (inverse-freq rare-class only, no mIoU signal), "
                             "invfreq_invmiou (inverse-freq × (1 - mIoU), boost-bad variant)")
    parser.add_argument("--loss", type=str, default="focaldice",
                        choices=["focaldice", "unifiedfocal", "tversky", "crossrecall",
                                 "recall", "recall_slice", "fciou"],
                        help="Supervised loss: focaldice (default), unifiedfocal, "
                             "tversky (asymmetric), crossrecall (FD + per-crossline recall), "
                             "recall (Tian 2021 FNR-weighted CE, batch-level), "
                             "recall_slice (Tian variant with per-slice FNR, novelty ablation), "
                             "fciou (MAKE 2023 focal IoU)")
    parser.add_argument("--lambda_crl", type=float, default=1.0,
                        help="Weight on crossline-recall term (only if --loss=crossrecall)")
    parser.add_argument("--force_rare_client", action="store_true",
                        help="Biased client sampling: guarantee ≥1 client with rare_fraction>0 "
                             "in each round (remaining slots sampled uniformly from the rest).")
    args = parser.parse_args()

    num_clients = args.num_clients
    split_mode = args.split
    num_rounds = args.num_rounds
    local_epochs = args.local_epochs
    batch_size = args.batch_size
    lr = args.lr
    seed = args.seed
    sample_ratio = args.sample_ratio
    use_class_weights = args.class_weights
    agg_strategy = args.agg_strategy

    sr_tag = f"_sr{sample_ratio}" if sample_ratio < 1.0 else ""
    cw_tag = "_cw" if use_class_weights else ""
    agg_tag = f"_agg{agg_strategy}" if agg_strategy != "equal" else ""
    le_tag = f"_le{local_epochs}" if local_epochs != LOCAL_EPOCHS else ""
    loss_tag = f"_loss{args.loss}" if args.loss != "focaldice" else ""
    frc_tag = "_frc" if args.force_rare_client else ""
    seed_tag = f"_s{seed}"
    exp_name = f"fedavg_{split_mode}_{num_clients}c_{num_rounds}r{sr_tag}{cw_tag}{agg_tag}{le_tag}{loss_tag}{frc_tag}{seed_tag}"
    save_dir = os.path.join(PROJECT_ROOT, "results", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # ── Startup ─────────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"FedAvg — Parihaka | {split_mode.upper()} | {num_clients} clients")
    print("=" * 70)
    print(f"Parihaka pipeline root : {PARIHAKA_ROOT}")
    print(f"Split mode             : {split_mode}")
    print(f"Num clients            : {num_clients}")
    print(f"Sample ratio           : {sample_ratio}")
    print(f"Aggregation strategy   : {agg_strategy}")
    print(f"Class weights          : {use_class_weights}")
    print(f"Loss function          : {args.loss}")
    print(f"Num rounds             : {num_rounds}")
    print(f"Local epochs           : {local_epochs}")
    print(f"Batch size             : {batch_size}")
    print(f"Learning rate          : {lr}")
    print(f"Device                 : {DEVICE}")
    print(f"Save directory         : {save_dir}")
    print("=" * 70)

    for p in [TRAIN_SEISMIC, TRAIN_LABELS, TEST1_SEISMIC, TEST1_LABELS, TEST2_SEISMIC, TEST2_LABELS]:
        assert os.path.exists(p), f"Data file not found: {p}"
    print("[OK] All data files found.\n")

    # ── wandb ───────────────────────────────────────────────────────────────
    wandb.init(
        project="FL-Seismic",
        group=f"fedavg_{split_mode}",
        name=exp_name,
        config={
            "algorithm": "FedAvg",
            "split_mode": split_mode,
            "num_clients": num_clients,
            "sample_ratio": sample_ratio,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "loss": args.loss,
            "lambda_crl": args.lambda_crl if args.loss == "crossrecall" else None,
            "focal_gamma": 2.0,
            "class_weights": use_class_weights,
            "agg_strategy": agg_strategy,
            "model": "UNet (2D)",
            "num_classes": NUM_CLASSES,
            "dataset": "Parihaka",
            "seed": seed,
        },
    )

    # ── Seed ────────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Load data ───────────────────────────────────────────────────────────
    print("Loading training data...")
    train_seismic = load_and_normalize(TRAIN_SEISMIC)
    train_labels = np.load(TRAIN_LABELS)
    num_crosslines = train_seismic.shape[1]
    print(f"  Training cube: {train_seismic.shape} (inline, crossline, depth)")
    print(f"  Unique labels: {np.unique(train_labels)}")

    # ── Partition ───────────────────────────────────────────────────────────
    if split_mode == "noniid":
        partitions = partition_noniid(num_crosslines, num_clients)
    else:
        partitions = partition_iid(num_crosslines, num_clients, rng)

    # Log class distribution per client for analysis
    print(f"\nPartitioning {num_crosslines} crosslines ({split_mode.upper()}) into {num_clients} clients:")
    client_class_dists = {}
    for c, idxs in enumerate(partitions):
        client_labels = train_labels[:, idxs, :].flatten()
        class_counts = {int(k): int(v) for k, v in zip(*np.unique(client_labels, return_counts=True))}
        total = sum(class_counts.values())
        class_pcts = {k: f"{100*v/total:.1f}%" for k, v in class_counts.items()}
        client_class_dists[c] = class_pcts
        print(f"  Client {c}: {len(idxs)} crosslines | class dist: {class_pcts}")

    # Log partition info to wandb
    wandb.config.update({"client_class_distributions": client_class_dists})

    # Precompute class info for weighted aggregation
    client_info = compute_client_class_info(train_labels, partitions)
    if agg_strategy != "equal":
        print(f"\n  Aggregation strategy: {agg_strategy}")
        for c, info in enumerate(client_info):
            print(f"    Client {c}: {info['num_classes']} classes, rare_frac={info['rare_fraction']:.4f}")

    client_loaders = build_client_loaders(train_seismic, train_labels, partitions, batch_size)

    # ── Test loaders ────────────────────────────────────────────────────────
    print("\nLoading test sets...")
    test_loader1, test_labels1 = build_test_loader(TEST1_SEISMIC, TEST1_LABELS)
    test_loader2, test_labels2 = build_test_loader(TEST2_SEISMIC, TEST2_LABELS)
    print(f"  Test1: {test_labels1.shape}")
    print(f"  Test2: {test_labels2.shape}")

    # ── Model ───────────────────────────────────────────────────────────────
    print("\nInitializing global UNet model (random weights)...")
    global_model = UNet(in_channels=1, num_classes=NUM_CLASSES, bilinear=False)
    print(f"  Parameters: {sum(p.numel() for p in global_model.parameters()):,}")

    # ── Class weights / loss selection ──────────────────────────────────
    RARE_CLASSES = [4, 5]
    if use_class_weights:
        unique, counts = np.unique(train_labels, return_counts=True)
        freqs = counts / counts.sum()
        inv_freqs = 1.0 / freqs
        alpha = (inv_freqs / inv_freqs.sum() * NUM_CLASSES).tolist()
        print(f"  Class weights (inv-freq): {['%.4f' % a for a in alpha]}")
    else:
        alpha = None

    if args.loss == "focaldice":
        criterion = FocalDiceLoss(focal_weight=1.0, dice_weight=1.0, gamma=2.0, alpha=alpha)
    elif args.loss == "unifiedfocal":
        # Asymmetric alpha/beta: rare classes get heavy FN penalty (beta high).
        alpha_pc = [0.5] * NUM_CLASSES
        beta_pc = [0.5] * NUM_CLASSES
        for rc in RARE_CLASSES:
            alpha_pc[rc] = 0.3
            beta_pc[rc] = 0.7
        criterion = UnifiedFocalLoss(
            alpha_per_class=alpha_pc, beta_per_class=beta_pc,
            gamma_ce=2.0, gamma_t=2.0, lambda_ce=0.5,
        )
        print(f"  Unified Focal loss: alpha={alpha_pc}, beta={beta_pc}")
    elif args.loss == "tversky":
        alpha_pc = [0.5] * NUM_CLASSES
        beta_pc = [0.5] * NUM_CLASSES
        for rc in RARE_CLASSES:
            alpha_pc[rc] = 0.3
            beta_pc[rc] = 0.7
        criterion = AsymmetricTverskyLoss(
            alpha_per_class=alpha_pc, beta_per_class=beta_pc, gamma=2.0,
        )
        print(f"  Asymmetric Tversky loss: alpha={alpha_pc}, beta={beta_pc}")
    elif args.loss == "crossrecall":
        criterion = CompoundCrosslineRecallLoss(
            rare_classes=RARE_CLASSES, lambda_crl=args.lambda_crl,
            gamma=2.0, alpha=alpha,
        )
        print(f"  Crossline-recall loss: rare={RARE_CLASSES}, lambda={args.lambda_crl}")
    elif args.loss == "recall":
        criterion = RecallLoss(num_classes=NUM_CLASSES)
        print(f"  Recall Loss (Tian 2021): FNR-weighted CE, num_classes={NUM_CLASSES}")
    elif args.loss == "recall_slice":
        criterion = RecallLossPerSlice(num_classes=NUM_CLASSES)
        print(f"  Recall Loss per-slice (novelty ablation): per-2D-slice FNR-weighted CE")
    elif args.loss == "fciou":
        criterion = FCIoULoss(gamma=2.0)
        print(f"  FCIoU Loss (MAKE 2023): focal per-class IoU, gamma=2.0")
    else:
        raise ValueError(f"Unknown --loss {args.loss}")
    best_miou = 0.0
    history = []

    # ── FedAvg Loop ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Starting Federated Averaging ({split_mode.upper()}, {num_clients} clients)")
    print("=" * 70 + "\n")

    for round_idx in range(num_rounds):
        round_start = time.time()
        print(f"--- Round {round_idx + 1}/{num_rounds} ---")

        global_weights = copy.deepcopy(global_model.state_dict())
        client_weights = []

        # Client subsampling
        if sample_ratio < 1.0:
            n_sampled = max(1, int(num_clients * sample_ratio))
            if args.force_rare_client:
                rare_pool = [c for c in range(num_clients) if client_info[c]["rare_fraction"] > 0]
                other_pool = [c for c in range(num_clients) if client_info[c]["rare_fraction"] == 0]
                if len(rare_pool) == 0:
                    selected = sorted(rng.choice(num_clients, n_sampled, replace=False).tolist())
                    print(f"  [force_rare_client] no rare clients exist — falling back to uniform")
                else:
                    forced = int(rng.choice(rare_pool))
                    n_remaining = n_sampled - 1
                    remaining_pool = [c for c in range(num_clients) if c != forced]
                    if n_remaining > 0:
                        extras = rng.choice(remaining_pool, n_remaining, replace=False).tolist()
                    else:
                        extras = []
                    selected = sorted([forced] + extras)
                    print(f"  [force_rare_client] forced={forced} (rare_frac="
                          f"{client_info[forced]['rare_fraction']:.4f}); "
                          f"sampled {n_sampled}/{num_clients}: {selected}")
            else:
                selected = sorted(rng.choice(num_clients, n_sampled, replace=False).tolist())
                print(f"  Sampled {n_sampled}/{num_clients} clients: {selected}")
        else:
            selected = list(range(num_clients))

        client_models_round = []
        client_class_ious_round = []
        needs_local_iou = agg_strategy in ("rare_miou", "invfreq_miou", "invfreq_invmiou")
        for c in selected:
            local_model = UNet(in_channels=1, num_classes=NUM_CLASSES, bilinear=False)
            local_model.load_state_dict(global_weights)
            updated_weights = local_train(
                model=local_model, loader=client_loaders[c], criterion=criterion,
                device=DEVICE, local_epochs=local_epochs, lr=lr, weight_decay=WEIGHT_DECAY,
            )
            client_weights.append(updated_weights)
            if agg_strategy == "accuracy":
                local_model.load_state_dict(updated_weights)
                client_models_round.append(local_model)
            if needs_local_iou:
                local_model.load_state_dict(updated_weights)
                ious = compute_local_class_iou(local_model, client_loaders[c], DEVICE)
                client_class_ious_round.append(ious)
            print(f"  Client {c} training complete.")

        if agg_strategy == "equal":
            aggregated_weights = fedavg_aggregate(client_weights)
        else:
            agg_w = get_agg_weights(
                agg_strategy, selected, client_info,
                client_models=client_models_round if agg_strategy == "accuracy" else None,
                test_loader=test_loader1 if agg_strategy == "accuracy" else None,
                test_labels=test_labels1 if agg_strategy == "accuracy" else None,
                device=DEVICE,
                client_class_ious=client_class_ious_round if needs_local_iou else None,
            )
            w_norm = np.array(agg_w) / np.sum(agg_w)
            print(f"  Agg weights ({agg_strategy}): {['%.3f' % w for w in w_norm]}")
            aggregated_weights = weighted_aggregate(client_weights, agg_w)
        global_model.load_state_dict(aggregated_weights)
        print("  Aggregation complete.")

        miou1, class_iou1 = evaluate_global(global_model, test_loader1, test_labels1, DEVICE)
        miou2, class_iou2 = evaluate_global(global_model, test_loader2, test_labels2, DEVICE)
        avg_miou = (miou1 + miou2) / 2.0
        avg_class_iou = (class_iou1 + class_iou2) / 2.0
        round_time = time.time() - round_start

        print(f"  Test1 mIoU: {miou1:.4f}  |  Test2 mIoU: {miou2:.4f}  |  Avg mIoU: {avg_miou:.4f}")
        print(f"  Per-class IoU (avg): {['%.4f' % v for v in avg_class_iou]}")
        print(f"  Round time: {round_time:.1f}s")

        # wandb logging
        log_dict = {
            "round": round_idx + 1,
            "global/miou_test1": miou1,
            "global/miou_test2": miou2,
            "global/miou_avg": avg_miou,
            "global/round_time_s": round_time,
        }
        for i in range(NUM_CLASSES):
            log_dict[f"per_class_iou_test1/class_{i}"] = class_iou1[i]
            log_dict[f"per_class_iou_test2/class_{i}"] = class_iou2[i]
            log_dict[f"per_class_iou_avg/class_{i}"] = avg_class_iou[i]
        wandb.log(log_dict, step=round_idx + 1)

        if avg_miou > best_miou:
            best_miou = avg_miou
            torch.save(global_model.state_dict(), os.path.join(save_dir, "best_global_model.pth"))
            print(f"  ** New best model saved (mIoU={best_miou:.4f}) **")

        torch.save(global_model.state_dict(), os.path.join(save_dir, f"global_round_{round_idx}.pth"))

        history.append({
            "round": round_idx + 1,
            "miou_test1": miou1, "miou_test2": miou2, "avg_miou": avg_miou,
            "per_class_iou_test1": class_iou1.tolist(),
            "per_class_iou_test2": class_iou2.tolist(),
            "time_s": round_time,
        })
        print()

    # ── Summary ─────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Training complete!")
    print(f"Best average mIoU: {best_miou:.4f}")
    print(f"Best model saved to: {os.path.join(save_dir, 'best_global_model.pth')}")
    print("=" * 70)

    np.save(os.path.join(save_dir, "training_history.npy"), history, allow_pickle=True)

    wandb.run.summary["best_avg_miou"] = best_miou
    artifact = wandb.Artifact(f"best_model_{exp_name}", type="model")
    artifact.add_file(os.path.join(save_dir, "best_global_model.pth"))
    wandb.log_artifact(artifact)
    wandb.finish()

    print(f"\nRound | Test1 mIoU | Test2 mIoU | Avg mIoU | Per-Class IoU (avg)")
    print("-" * 90)
    for h in history:
        cls = " ".join([f"{v:.3f}" for v in ((np.array(h['per_class_iou_test1']) + np.array(h['per_class_iou_test2'])) / 2)])
        print(f"  {h['round']:2d}  |   {h['miou_test1']:.4f}   |   {h['miou_test2']:.4f}   |  {h['avg_miou']:.4f}  | [{cls}]")


if __name__ == "__main__":
    main()
