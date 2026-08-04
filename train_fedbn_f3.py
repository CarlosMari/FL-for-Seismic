"""
FedBN — Federated Learning with Local Batch Normalization (F3)
==============================================================
Like FedAvg, but BatchNorm parameters are kept LOCAL to each client.
Only conv/linear weights are averaged during aggregation.

Usage:
    python train_fedbn_f3.py --num_clients 3 --split noniid
    python train_fedbn_f3.py --num_clients 20 --split iid
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

F3_SEGY = os.path.join(PROJECT_ROOT, "..", "OCT_SEISMIC_PIPELINE", "data", "f3", "f3_train.segy")
F3_LABELS = os.path.join(PROJECT_ROOT, "..", "OCT_SEISMIC_PIPELINE", "data", "f3", "f3_train_labels.npy")

from train_tools.models.unet import UNet
from train_tools.preprocessing.seismic.datasets import InlineLoader

# ── Fixed hyperparameters ───────────────────────────────────────────────────
NUM_ROUNDS       = 20
LOCAL_EPOCHS     = 3
BATCH_SIZE       = 4
LR               = 1e-3
WEIGHT_DECAY     = 1e-4
NUM_CLASSES      = 6
TEST_SPLIT_RATIO = 0.2
DEVICE           = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED             = 42

N_INLINES    = 401
N_CROSSLINES = 701
N_DEPTH      = 255


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
    def __init__(self, focal_weight=1.0, dice_weight=1.0, gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma)
        self.dice = DiceLoss()
        self.fw = focal_weight
        self.dw = dice_weight

    def forward(self, logits, targets):
        return self.fw * self.focal(logits, targets) + self.dw * self.dice(logits, targets)


# ── BN key detection ──────────────────────────────────────────────────────
def is_bn_key(key):
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "double_conv" and i + 1 < len(parts) and parts[i + 1] in ("1", "4"):
            return True
    return False


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
def load_f3_segy(segy_path):
    import segyio
    with segyio.open(segy_path, ignore_geometry=True) as f:
        traces = np.array([f.trace[i] for i in range(f.tracecount)])
    cube = traces.reshape(N_INLINES, N_CROSSLINES, N_DEPTH)
    cube = (cube - cube.min()) / (cube.max() - cube.min() + 1e-8)
    return cube


def build_client_loaders(seismic, labels, partitions, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    loaders = []
    for c, idxs in enumerate(partitions):
        dataset = InlineLoader(
            seismic_cube=seismic, label_cube=labels,
            inline_inds=idxs, train_status=True, transform=transform,
        )
        loader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        loaders.append(loader)
        if len(idxs) <= 10:
            idx_str = str(idxs)
        else:
            idx_str = f"[{idxs[0]}..{idxs[-1]}]"
        print(f"  Client {c}: {len(idxs)} crosslines {idx_str}")
    return loaders


def build_test_loader(seismic, labels, crossline_idxs, batch_size=1):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = InlineLoader(
        seismic_cube=seismic, label_cube=labels,
        inline_inds=crossline_idxs, train_status=False, transform=transform,
    )
    loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    gt_labels = labels[:, crossline_idxs, :]
    return loader, gt_labels


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


# ── FedBN aggregation ─────────────────────────────────────────────────────
def fedbn_aggregate(state_dicts):
    n = len(state_dicts)
    avg = copy.deepcopy(state_dicts[0])
    for key in avg.keys():
        if is_bn_key(key):
            continue
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
    parser = argparse.ArgumentParser(description="FedBN on F3")
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--split", type=str, default="noniid", choices=["iid", "noniid"])
    parser.add_argument("--num_rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--local_epochs", type=int, default=LOCAL_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    num_clients = args.num_clients
    split_mode = args.split
    num_rounds = args.num_rounds
    local_epochs = args.local_epochs
    batch_size = args.batch_size
    lr = args.lr
    seed = args.seed

    exp_name = f"fedbn_f3_{split_mode}_{num_clients}c_{num_rounds}r"
    save_dir = os.path.join(PROJECT_ROOT, "results", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print(f"FedBN — F3 Netherlands | {split_mode.upper()} | {num_clients} clients")
    print("=" * 70)
    print(f"Split mode             : {split_mode}")
    print(f"Num clients            : {num_clients}")
    print(f"Num rounds             : {num_rounds}")
    print(f"Local epochs           : {local_epochs}")
    print(f"Batch size             : {batch_size}")
    print(f"Learning rate          : {lr}")
    print(f"Device                 : {DEVICE}")
    print(f"Save directory         : {save_dir}")
    print("=" * 70)

    for p in [F3_SEGY, F3_LABELS]:
        assert os.path.exists(p), f"Data file not found: {p}"
    print("[OK] All data files found.\n")

    wandb.init(
        project="FL-Seismic",
        group=f"fedbn_f3_{split_mode}",
        name=exp_name,
        config={
            "algorithm": "FedBN",
            "dataset": "F3_Netherlands",
            "split_mode": split_mode,
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "loss": "FocalLoss + DiceLoss",
            "focal_gamma": 2.0,
            "model": "UNet (2D)",
            "num_classes": NUM_CLASSES,
            "test_split_ratio": TEST_SPLIT_RATIO,
            "seed": seed,
        },
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Loading F3 seismic cube from SEGY...")
    seismic = load_f3_segy(F3_SEGY)
    labels = np.load(F3_LABELS)
    print(f"  Cube shape: {seismic.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique labels: {np.unique(labels)}")

    total_crosslines = seismic.shape[1]
    n_test = int(total_crosslines * TEST_SPLIT_RATIO)
    n_train = total_crosslines - n_test
    test_crosslines = list(range(n_train, total_crosslines))

    print(f"\n  Train crosslines: [0, {n_train}) -> {n_train}")
    print(f"  Test crosslines:  [{n_train}, {total_crosslines}) -> {n_test}")

    if split_mode == "noniid":
        partitions = partition_noniid(n_train, num_clients)
    else:
        partitions = partition_iid(n_train, num_clients, rng)

    print(f"\nPartitioning {n_train} train crosslines ({split_mode.upper()}) into {num_clients} clients:")
    client_class_dists = {}
    for c, idxs in enumerate(partitions):
        client_labels = labels[:, idxs, :].flatten()
        class_counts = {int(k): int(v) for k, v in zip(*np.unique(client_labels, return_counts=True))}
        total = sum(class_counts.values())
        class_pcts = {k: f"{100*v/total:.1f}%" for k, v in class_counts.items()}
        client_class_dists[c] = class_pcts
        print(f"  Client {c}: {len(idxs)} crosslines | class dist: {class_pcts}")
    wandb.config.update({"client_class_distributions": client_class_dists})

    client_loaders = build_client_loaders(seismic, labels, partitions, batch_size)

    print("\nBuilding global test loader...")
    test_loader, test_labels = build_test_loader(seismic, labels, test_crosslines)
    print(f"  Test labels shape: {test_labels.shape}")

    print("\nInitializing global UNet model (random weights)...")
    global_model = UNet(in_channels=1, num_classes=NUM_CLASSES, bilinear=False)
    print(f"  Parameters: {sum(p.numel() for p in global_model.parameters()):,}")

    all_keys = list(global_model.state_dict().keys())
    bn_keys = [k for k in all_keys if is_bn_key(k)]
    non_bn_keys = [k for k in all_keys if not is_bn_key(k)]
    print(f"  BN keys (local):      {len(bn_keys)}")
    print(f"  Non-BN keys (shared): {len(non_bn_keys)}")

    criterion = FocalDiceLoss(focal_weight=1.0, dice_weight=1.0, gamma=2.0)
    best_miou = 0.0
    history = []
    client_bn_states = [None] * num_clients

    print("\n" + "=" * 70)
    print(f"Starting FedBN ({split_mode.upper()}, {num_clients} clients)")
    print("=" * 70 + "\n")

    for round_idx in range(num_rounds):
        round_start = time.time()
        print(f"--- Round {round_idx + 1}/{num_rounds} ---")

        global_weights = copy.deepcopy(global_model.state_dict())
        client_weights = []

        for c in range(num_clients):
            local_model = UNet(in_channels=1, num_classes=NUM_CLASSES, bilinear=False)
            local_model.load_state_dict(global_weights, strict=False)

            if client_bn_states[c] is not None:
                local_sd = local_model.state_dict()
                for key in bn_keys:
                    local_sd[key] = client_bn_states[c][key]
                local_model.load_state_dict(local_sd)

            updated_weights = local_train(
                model=local_model, loader=client_loaders[c], criterion=criterion,
                device=DEVICE, local_epochs=local_epochs, lr=lr, weight_decay=WEIGHT_DECAY,
            )
            client_bn_states[c] = {k: updated_weights[k].clone() for k in bn_keys}
            client_weights.append(updated_weights)
            print(f"  Client {c} training complete.")

        aggregated_weights = fedbn_aggregate(client_weights)
        global_model.load_state_dict(aggregated_weights)
        print("  Aggregation complete (BN layers excluded).")

        # Evaluate with each client's BN and average
        eval_model = UNet(in_channels=1, num_classes=NUM_CLASSES, bilinear=False)
        eval_sd = copy.deepcopy(aggregated_weights)

        client_mious = []
        client_class_ious = []

        for c in range(num_clients):
            for key in bn_keys:
                eval_sd[key] = client_bn_states[c][key]
            eval_model.load_state_dict(eval_sd)
            miou_c, ciou_c = evaluate_global(eval_model, test_loader, test_labels, DEVICE)
            client_mious.append(miou_c)
            client_class_ious.append(ciou_c)

        miou = np.mean(client_mious)
        avg_class_iou = np.mean(client_class_ious, axis=0)
        best_client_miou = max(client_mious)
        round_time = time.time() - round_start

        print(f"  mIoU (avg across clients): {miou:.4f}")
        print(f"  Per-class IoU (avg): {['%.4f' % v for v in avg_class_iou]}")
        print(f"  Best single-client mIoU: {best_client_miou:.4f}")
        print(f"  Per-client mIoU: {['%.4f' % m for m in client_mious]}")
        print(f"  Round time: {round_time:.1f}s")

        log_dict = {
            "round": round_idx + 1,
            "global/miou": miou,
            "global/best_client_miou": best_client_miou,
            "global/round_time_s": round_time,
        }
        for i in range(NUM_CLASSES):
            log_dict[f"per_class_iou/class_{i}"] = avg_class_iou[i]
        for c in range(num_clients):
            log_dict[f"client_miou/client_{c}"] = client_mious[c]
        wandb.log(log_dict, step=round_idx + 1)

        if miou > best_miou:
            best_miou = miou
            torch.save(aggregated_weights, os.path.join(save_dir, "best_global_model.pth"))
            torch.save(client_bn_states, os.path.join(save_dir, "best_client_bn_states.pth"))
            print(f"  ** New best model saved (mIoU={best_miou:.4f}) **")

        history.append({
            "round": round_idx + 1,
            "miou": miou,
            "per_class_iou": avg_class_iou.tolist(),
            "best_client_miou": best_client_miou,
            "client_mious": client_mious,
            "time_s": round_time,
        })
        print()

    print("=" * 70)
    print("Training complete!")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"Best model saved to: {os.path.join(save_dir, 'best_global_model.pth')}")
    print("=" * 70)

    np.save(os.path.join(save_dir, "training_history.npy"), history, allow_pickle=True)

    wandb.run.summary["best_miou"] = best_miou
    artifact = wandb.Artifact(f"best_model_{exp_name}", type="model")
    artifact.add_file(os.path.join(save_dir, "best_global_model.pth"))
    wandb.log_artifact(artifact)
    wandb.finish()

    print(f"\nRound | mIoU   | Best Client | Per-Class IoU")
    print("-" * 80)
    for h in history:
        cls = " ".join([f"{v:.3f}" for v in h['per_class_iou']])
        print(f"  {h['round']:2d}  | {h['miou']:.4f} |    {h['best_client_miou']:.4f}   | [{cls}]")


if __name__ == "__main__":
    main()
