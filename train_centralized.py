"""
Centralized (Non-Federated) Baseline on Parihaka and F3 Datasets
================================================================
Trains UNet on ALL training data with no federation, providing the
upper-bound reference for federated experiments.

Usage:
    python train_centralized.py --dataset parihaka
    python train_centralized.py --dataset f3
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.transforms import transforms
from sklearn.metrics import jaccard_score
import wandb

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from train_tools.models.unet import UNet
from train_tools.preprocessing.seismic.datasets import InlineLoader

# ── Paths ───────────────────────────────────────────────────────────────────
TRAIN_SEISMIC = os.path.join(PROJECT_ROOT, "datasets", "train", "train_seismic.npy")
TRAIN_LABELS  = os.path.join(PROJECT_ROOT, "datasets", "train", "train_labels.npy")
TEST1_SEISMIC = os.path.join(PROJECT_ROOT, "datasets", "test_once", "test1_seismic.npy")
TEST1_LABELS  = os.path.join(PROJECT_ROOT, "datasets", "test_once", "test1_labels.npy")
TEST2_SEISMIC = os.path.join(PROJECT_ROOT, "datasets", "test_once", "test2_seismic.npy")
TEST2_LABELS  = os.path.join(PROJECT_ROOT, "datasets", "test_once", "test2_labels.npy")
F3_SEGY   = os.path.join(PROJECT_ROOT, "..", "OCT_SEISMIC_PIPELINE", "data", "f3", "f3_train.segy")
F3_LABELS = os.path.join(PROJECT_ROOT, "..", "OCT_SEISMIC_PIPELINE", "data", "f3", "f3_train_labels.npy")

# ── Hyperparameters (matched to federated experiments) ──────────────────────
# In federated: 20 rounds x 3 local epochs = 60 total epochs of training
# Centralized gets the same total epochs for fair comparison
TOTAL_EPOCHS = 60
BATCH_SIZE   = 4
LR           = 1e-3
WEIGHT_DECAY = 1e-4
NUM_CLASSES  = 6
DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED         = 42

N_INLINES    = 401
N_CROSSLINES = 701
N_DEPTH      = 255


# ── Loss functions ──────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
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
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.focal(logits, targets) + self.dice(logits, targets)


# ── Data helpers ────────────────────────────────────────────────────────────
def load_and_normalize(path):
    arr = np.load(path)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def load_f3_segy(segy_path):
    import segyio
    with segyio.open(segy_path, ignore_geometry=True) as f:
        traces = np.array([f.trace[i] for i in range(f.tracecount)])
    cube = traces.reshape(N_INLINES, N_CROSSLINES, N_DEPTH)
    cube = (cube - cube.min()) / (cube.max() - cube.min() + 1e-8)
    return cube


def build_loader(seismic, labels, crossline_idxs, train, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = InlineLoader(
        seismic_cube=seismic, label_cube=labels,
        inline_inds=crossline_idxs, train_status=train, transform=transform,
    )
    loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True
    )
    return loader


# ── Evaluation ──────────────────────────────────────────────────────────────
def evaluate(model, test_loader, test_labels, device):
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


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Centralized baseline")
    parser.add_argument("--dataset", type=str, default="parihaka", choices=["parihaka", "f3"])
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval_every", type=int, default=3,
                        help="Evaluate every N epochs (default: 3, matching FL round granularity)")
    args = parser.parse_args()

    dataset_name = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    seed = args.seed
    eval_every = args.eval_every

    exp_name = f"centralized_{dataset_name}_{epochs}ep"
    save_dir = os.path.join(PROJECT_ROOT, "results", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print(f"Centralized Baseline — {dataset_name.upper()}")
    print("=" * 70)
    print(f"Epochs                 : {epochs}")
    print(f"Batch size             : {batch_size}")
    print(f"Learning rate          : {lr}")
    print(f"Eval every             : {eval_every} epochs")
    print(f"Device                 : {DEVICE}")
    print(f"Save directory         : {save_dir}")
    print("=" * 70)

    # ── Seed ────────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Load data ───────────────────────────────────────────────────────────
    if dataset_name == "parihaka":
        print("Loading Parihaka data...")
        train_seismic = load_and_normalize(TRAIN_SEISMIC)
        train_labels = np.load(TRAIN_LABELS)
        num_train_crosslines = train_seismic.shape[1]
        train_loader = build_loader(train_seismic, train_labels,
                                    list(range(num_train_crosslines)), True, batch_size)

        test_seismic1 = load_and_normalize(TEST1_SEISMIC)
        test_labels1 = np.load(TEST1_LABELS)
        test_loader1 = build_loader(test_seismic1, test_labels1,
                                    list(range(test_seismic1.shape[1])), False, 1)

        test_seismic2 = load_and_normalize(TEST2_SEISMIC)
        test_labels2 = np.load(TEST2_LABELS)
        test_loader2 = build_loader(test_seismic2, test_labels2,
                                    list(range(test_seismic2.shape[1])), False, 1)

        print(f"  Train: {train_seismic.shape} ({num_train_crosslines} crosslines)")
        print(f"  Test1: {test_labels1.shape}, Test2: {test_labels2.shape}")
        has_two_tests = True

    else:  # f3
        print("Loading F3 data from SEGY...")
        seismic = load_f3_segy(F3_SEGY)
        labels = np.load(F3_LABELS)
        total_crosslines = seismic.shape[1]
        n_test = int(total_crosslines * 0.2)
        n_train = total_crosslines - n_test

        train_loader = build_loader(seismic, labels, list(range(n_train)), True, batch_size)

        test_crosslines = list(range(n_train, total_crosslines))
        test_loader1 = build_loader(seismic, labels, test_crosslines, False, 1)
        test_labels1 = labels[:, test_crosslines, :]

        print(f"  Cube: {seismic.shape}")
        print(f"  Train: {n_train} crosslines, Test: {n_test} crosslines")
        has_two_tests = False

    # ── wandb ───────────────────────────────────────────────────────────────
    wandb.init(
        project="FL-Seismic",
        group=f"centralized",
        name=exp_name,
        config={
            "algorithm": "Centralized",
            "dataset": dataset_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "loss": "FocalLoss + DiceLoss",
            "model": "UNet (2D)",
            "num_classes": NUM_CLASSES,
            "seed": seed,
        },
    )

    # ── Model ───────────────────────────────────────────────────────────────
    print("\nInitializing UNet model...")
    model = UNet(in_channels=1, num_classes=NUM_CLASSES, bilinear=False)
    model.to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = FocalDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    best_miou = 0.0
    history = []

    print("\n" + "=" * 70)
    print(f"Starting centralized training ({epochs} epochs)")
    print("=" * 70 + "\n")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, targets, _ in train_loader:
            images = images.to(DEVICE, dtype=torch.float)
            targets = targets.to(DEVICE, dtype=torch.long)
            logits, _ = model(images)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        epoch_time = time.time() - epoch_start

        if epoch % eval_every == 0 or epoch == epochs:
            miou1, class_iou1 = evaluate(model, test_loader1, test_labels1, DEVICE)

            if has_two_tests:
                miou2, class_iou2 = evaluate(model, test_loader2, test_labels2, DEVICE)
                avg_miou = (miou1 + miou2) / 2.0
                avg_class_iou = (class_iou1 + class_iou2) / 2.0
                print(f"Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | "
                      f"Test1={miou1:.4f} Test2={miou2:.4f} Avg={avg_miou:.4f} | "
                      f"time={epoch_time:.1f}s")
                print(f"  Per-class (avg): {['%.4f' % v for v in avg_class_iou]}")

                log_dict = {
                    "epoch": epoch, "train/loss": avg_loss,
                    "global/miou_test1": miou1, "global/miou_test2": miou2,
                    "global/miou_avg": avg_miou, "global/epoch_time_s": epoch_time,
                }
                for i in range(NUM_CLASSES):
                    log_dict[f"per_class_iou_avg/class_{i}"] = avg_class_iou[i]
                wandb.log(log_dict, step=epoch)

                eval_miou = avg_miou
                eval_class_iou = avg_class_iou
            else:
                avg_miou = miou1
                print(f"Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | "
                      f"mIoU={miou1:.4f} | time={epoch_time:.1f}s")
                print(f"  Per-class: {['%.4f' % v for v in class_iou1]}")

                log_dict = {
                    "epoch": epoch, "train/loss": avg_loss,
                    "global/miou": miou1, "global/epoch_time_s": epoch_time,
                }
                for i in range(NUM_CLASSES):
                    log_dict[f"per_class_iou/class_{i}"] = class_iou1[i]
                wandb.log(log_dict, step=epoch)

                eval_miou = miou1
                eval_class_iou = class_iou1

            if eval_miou > best_miou:
                best_miou = eval_miou
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                print(f"  ** New best model (mIoU={best_miou:.4f}) **")

            history.append({
                "epoch": epoch, "loss": avg_loss, "miou": eval_miou,
                "per_class_iou": eval_class_iou.tolist(), "time_s": epoch_time,
            })
        else:
            wandb.log({"epoch": epoch, "train/loss": avg_loss}, step=epoch)
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | time={epoch_time:.1f}s")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"Best model saved to: {os.path.join(save_dir, 'best_model.pth')}")
    print("=" * 70)

    np.save(os.path.join(save_dir, "training_history.npy"), history, allow_pickle=True)

    wandb.run.summary["best_miou"] = best_miou
    artifact = wandb.Artifact(f"best_model_{exp_name}", type="model")
    artifact.add_file(os.path.join(save_dir, "best_model.pth"))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
