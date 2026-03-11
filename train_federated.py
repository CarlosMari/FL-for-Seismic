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

    sr_tag = f"_sr{sample_ratio}" if sample_ratio < 1.0 else ""
    cw_tag = "_cw" if use_class_weights else ""
    exp_name = f"fedavg_{split_mode}_{num_clients}c_{num_rounds}r{sr_tag}{cw_tag}"
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
    print(f"Class weights          : {use_class_weights}")
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
            "loss": "FocalLoss + DiceLoss",
            "focal_gamma": 2.0,
            "class_weights": use_class_weights,
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

    # ── Class weights ────────────────────────────────────────────────────
    if use_class_weights:
        unique, counts = np.unique(train_labels, return_counts=True)
        freqs = counts / counts.sum()
        inv_freqs = 1.0 / freqs
        alpha = (inv_freqs / inv_freqs.sum() * NUM_CLASSES).tolist()
        print(f"  Class weights (inv-freq): {['%.4f' % a for a in alpha]}")
        criterion = FocalDiceLoss(focal_weight=1.0, dice_weight=1.0, gamma=2.0, alpha=alpha)
    else:
        criterion = FocalDiceLoss(focal_weight=1.0, dice_weight=1.0, gamma=2.0)
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
            selected = sorted(rng.choice(num_clients, n_sampled, replace=False).tolist())
            print(f"  Sampled {n_sampled}/{num_clients} clients: {selected}")
        else:
            selected = list(range(num_clients))

        for c in selected:
            local_model = UNet(in_channels=1, num_classes=NUM_CLASSES, bilinear=False)
            local_model.load_state_dict(global_weights)
            updated_weights = local_train(
                model=local_model, loader=client_loaders[c], criterion=criterion,
                device=DEVICE, local_epochs=local_epochs, lr=lr, weight_decay=WEIGHT_DECAY,
            )
            client_weights.append(updated_weights)
            print(f"  Client {c} training complete.")

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
