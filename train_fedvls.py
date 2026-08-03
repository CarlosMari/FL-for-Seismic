"""
FedVLS — Federated Learning with Vacant-class distillation & Logit Suppression
===============================================================================
Addresses the "vacant class" problem in label-skewed FL: clients missing
certain classes destroy global knowledge about them during local training.

Three-part local loss:
  L = L_cal + lambda * L_dis + L_logit

  L_cal:   Calibrated cross-entropy (softmax weighted by local class freqs)
  L_dis:   KL divergence on vacant-class logits (global model -> local model)
  L_logit: Logit suppression for non-label classes

Reference: Guo et al., "Exploring Vacant Classes in Label-Skewed Federated
Learning", AAAI 2024.

Usage:
    python train_fedvls.py --num_clients 3 --split noniid
    python train_fedvls.py --num_clients 20 --split noniid --lam 0.5
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


# ── Loss: DiceLoss (unchanged, used alongside calibrated CE) ───────────────
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


# ── FedVLS loss components ─────────────────────────────────────────────────
def calibrated_cross_entropy(logits, targets, class_freq):
    """
    L_cal: Calibrated CE — weight logits by local class frequency in softmax.
    logits: (B, C, H, W), targets: (B, H, W), class_freq: (C,) tensor on same device
    """
    # class_freq is p(c) for each class based on local data
    # L_cal = -log( p(y)*exp(f[y]) / sum_c p(c)*exp(f[c]) )
    # Equivalent to: CE with logits shifted by log(p(c))
    log_freq = torch.log(class_freq + 1e-10)  # (C,)
    # Add log_freq as bias to logits: calibrated_logits[c] = logits[c] + log(p(c))
    calibrated = logits + log_freq[None, :, None, None]  # broadcast to (B, C, H, W)
    return F.cross_entropy(calibrated, targets)


def vacant_class_distillation(logits_local, logits_global, vacant_mask):
    """
    L_dis: KL divergence between global and local model on vacant classes only.
    logits_local, logits_global: (B, C, H, W)
    vacant_mask: (C,) bool tensor — True for vacant classes
    """
    if not vacant_mask.any():
        return torch.tensor(0.0, device=logits_local.device)

    vc_idx = vacant_mask.nonzero(as_tuple=True)[0]

    # Softmax over the FULL class dim so the global model's "how present is
    # class c relative to other classes" signal is preserved. Selecting after
    # softmax avoids the |vacant|==1 degenerate case where softmax over a
    # length-1 dim is identically 1.0 and KL is structurally zero.
    local_log_probs_full = F.log_softmax(logits_local, dim=1)
    global_probs_full = F.softmax(logits_global, dim=1)

    kl_full = global_probs_full * (
        torch.log(global_probs_full + 1e-10) - local_log_probs_full
    )
    kl = kl_full[:, vc_idx, :, :].sum(dim=1).mean()
    return kl


def logit_suppression(logits, targets, class_freq):
    """
    L_logit: Penalize high logits for non-label classes.
    L_logit^c = log( E[ I(y!=c) * exp(f[c]) ] )
    L_logit = sum_c p(c) * L_logit^c
    """
    B, C, H, W = logits.shape
    # Reshape for computation: (B*H*W, C)
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)
    targets_flat = targets.reshape(-1)  # (N,)
    N = logits_flat.shape[0]

    total_loss = torch.tensor(0.0, device=logits.device)
    for c in range(C):
        # Indicator: y != c
        mask = (targets_flat != c).float()  # (N,)
        if mask.sum() == 0:
            continue
        # Max-subtracted: log(mean(exp(x))) == log(mean(exp(x - M))) + M.
        # Without this, exp() overflows to inf for logits above ~88 in float32.
        col = logits_flat[:, c]
        M = col.max().detach()
        mean_exp = (mask * torch.exp(col - M)).sum() / mask.sum()
        l_c = torch.log(mean_exp + 1e-10) + M
        total_loss = total_loss + class_freq[c] * l_c

    return total_loss


# ── Data partitioning ──────────────────────────────────────────────────────
def partition_noniid(num_crosslines, num_clients):
    chunk_size = num_crosslines // num_clients
    partitions = []
    for c in range(num_clients):
        start = c * chunk_size
        end = (c + 1) * chunk_size if c < num_clients - 1 else num_crosslines
        partitions.append(list(range(start, end)))
    return partitions


def partition_iid(num_crosslines, num_clients, rng):
    all_idxs = np.arange(num_crosslines)
    rng.shuffle(all_idxs)
    partitions = []
    chunk_size = num_crosslines // num_clients
    for c in range(num_clients):
        start = c * chunk_size
        end = (c + 1) * chunk_size if c < num_clients - 1 else num_crosslines
        partitions.append(sorted(all_idxs[start:end].tolist()))
    return partitions


# ── Data loading ────────────────────────────────────────────────────────────
def load_and_normalize(path):
    arr = np.load(path)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def build_client_loaders(train_seismic, train_labels, partitions, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    client_loaders = []
    for c, crossline_idxs in enumerate(partitions):
        dataset = InlineLoader(
            seismic_cube=train_seismic, label_cube=train_labels,
            inline_inds=crossline_idxs, train_status=True, transform=transform,
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


# ── FedAvg aggregation (same as baseline) ──────────────────────────────────
def fedavg_aggregate(state_dicts):
    n = len(state_dicts)
    avg = copy.deepcopy(state_dicts[0])
    for key in avg.keys():
        for i in range(1, n):
            avg[key] = avg[key] + state_dicts[i][key]
        avg[key] = avg[key] / n
    return avg


# ── FedVLS local training ─────────────────────────────────────────────────
def local_train_fedvls(model, global_model, loader, device, local_epochs, lr,
                       weight_decay, class_freq, vacant_mask, lam):
    """
    FedVLS local training with three-part loss:
      L = L_cal + DiceLoss + lam * L_dis + L_logit
    """
    model.train()
    model.to(device)
    global_model.eval()
    global_model.to(device)

    class_freq_t = torch.tensor(class_freq, dtype=torch.float, device=device)
    vacant_mask_t = torch.tensor(vacant_mask, dtype=torch.bool, device=device)

    dice_loss_fn = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(local_epochs):
        for images, targets, _ in loader:
            images = images.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)

            logits, _ = model(images)

            # L_cal: calibrated cross-entropy
            l_cal = calibrated_cross_entropy(logits, targets, class_freq_t)

            # DiceLoss (keep from original pipeline)
            l_dice = dice_loss_fn(logits, targets)

            # L_dis: vacant-class distillation from global model
            with torch.no_grad():
                global_logits, _ = global_model(images)
            l_dis = vacant_class_distillation(logits, global_logits, vacant_mask_t)

            # L_logit: logit suppression
            l_logit = logit_suppression(logits, targets, class_freq_t)

            loss = l_cal + l_dice + lam * l_dis + l_logit

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return copy.deepcopy(model.state_dict())


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FedVLS on Parihaka")
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--split", type=str, default="noniid", choices=["iid", "noniid"])
    parser.add_argument("--num_rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--local_epochs", type=int, default=LOCAL_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lam", type=float, default=0.1,
                        help="Weight for vacant-class distillation loss (default: 0.1)")
    args = parser.parse_args()

    num_clients = args.num_clients
    split_mode = args.split
    num_rounds = args.num_rounds
    local_epochs = args.local_epochs
    batch_size = args.batch_size
    lr = args.lr
    seed = args.seed
    lam = args.lam

    exp_name = f"fedvls_{split_mode}_{num_clients}c_{num_rounds}r_lam{lam}"
    save_dir = os.path.join(PROJECT_ROOT, "results", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print(f"FedVLS — Parihaka | {split_mode.upper()} | {num_clients} clients | lam={lam}")
    print("=" * 70)
    print(f"Split mode             : {split_mode}")
    print(f"Num clients            : {num_clients}")
    print(f"Lambda (distillation)  : {lam}")
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

    wandb.init(
        project="FL-Seismic",
        group=f"fedvls_{split_mode}",
        name=exp_name,
        config={
            "algorithm": "FedVLS",
            "split_mode": split_mode,
            "num_clients": num_clients,
            "lambda_dis": lam,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "loss": "CalibratedCE + DiceLoss + VacantDistill + LogitSuppression",
            "model": "UNet (2D)",
            "num_classes": NUM_CLASSES,
            "dataset": "Parihaka",
            "seed": seed,
        },
    )

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
    print(f"  Training cube: {train_seismic.shape}")

    # ── Partition ───────────────────────────────────────────────────────────
    if split_mode == "noniid":
        partitions = partition_noniid(num_crosslines, num_clients)
    else:
        partitions = partition_iid(num_crosslines, num_clients, rng)

    print(f"\nPartitioning {num_crosslines} crosslines ({split_mode.upper()}) into {num_clients} clients:")

    # Compute per-client class frequencies and vacant classes
    client_class_freqs = []
    client_vacant_masks = []
    client_class_dists = {}

    for c, idxs in enumerate(partitions):
        client_labels = train_labels[:, idxs, :].flatten()
        unique, counts = np.unique(client_labels, return_counts=True)
        total = counts.sum()

        # Build full frequency vector (0 for missing classes)
        freq = np.zeros(NUM_CLASSES, dtype=np.float32)
        for u, cnt in zip(unique, counts):
            freq[int(u)] = cnt / total

        # Vacant mask: True for classes with zero samples
        vacant = (freq == 0)

        client_class_freqs.append(freq)
        client_vacant_masks.append(vacant)

        class_pcts = {int(u): f"{100*cnt/total:.1f}%" for u, cnt in zip(unique, counts)}
        vacant_classes = [i for i in range(NUM_CLASSES) if vacant[i]]
        client_class_dists[c] = class_pcts
        print(f"  Client {c}: {len(idxs)} crosslines | dist: {class_pcts} | vacant: {vacant_classes}")

    wandb.config.update({"client_class_distributions": client_class_dists})

    client_loaders = build_client_loaders(train_seismic, train_labels, partitions, batch_size)

    # ── Test loaders ────────────────────────────────────────────────────────
    print("\nLoading test sets...")
    test_loader1, test_labels1 = build_test_loader(TEST1_SEISMIC, TEST1_LABELS)
    test_loader2, test_labels2 = build_test_loader(TEST2_SEISMIC, TEST2_LABELS)
    print(f"  Test1: {test_labels1.shape}")
    print(f"  Test2: {test_labels2.shape}")

    # ── Model ───────────────────────────────────────────────────────────────
    print("\nInitializing global UNet model...")
    global_model = UNet(in_channels=1, num_classes=NUM_CLASSES, bilinear=False)
    print(f"  Parameters: {sum(p.numel() for p in global_model.parameters()):,}")

    best_miou = 0.0
    history = []

    # ── FedVLS Loop ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Starting FedVLS ({split_mode.upper()}, {num_clients} clients, lam={lam})")
    print("=" * 70 + "\n")

    for round_idx in range(num_rounds):
        round_start = time.time()
        print(f"--- Round {round_idx + 1}/{num_rounds} ---")

        global_weights = copy.deepcopy(global_model.state_dict())
        client_weights = []

        # Keep a copy of global model for distillation
        global_model_frozen = UNet(in_channels=1, num_classes=NUM_CLASSES, bilinear=False)
        global_model_frozen.load_state_dict(global_weights)

        for c in range(num_clients):
            local_model = UNet(in_channels=1, num_classes=NUM_CLASSES, bilinear=False)
            local_model.load_state_dict(global_weights)

            updated_weights = local_train_fedvls(
                model=local_model,
                global_model=global_model_frozen,
                loader=client_loaders[c],
                device=DEVICE,
                local_epochs=local_epochs,
                lr=lr,
                weight_decay=WEIGHT_DECAY,
                class_freq=client_class_freqs[c],
                vacant_mask=client_vacant_masks[c],
                lam=lam,
            )
            client_weights.append(updated_weights)
            vc = [i for i in range(NUM_CLASSES) if client_vacant_masks[c][i]]
            print(f"  Client {c} done (vacant: {vc})")

        aggregated_weights = fedavg_aggregate(client_weights)
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
