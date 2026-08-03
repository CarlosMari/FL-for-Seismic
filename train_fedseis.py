"""
FedSeis — Prototype-Based FL for Seismic Facies Segmentation
=============================================================
Novel method combining:
  1. Server-held prototype slices (1 crossline/class, ~0.9% of training data)
     → Server extracts per-class feature prototypes each round and broadcasts.
     → Clients add a prototype alignment loss so features for class c are
       pulled toward prototype_c — including for classes the client has
       never seen (solves "data absence" at the feature level).
  2. Vacant-class distillation (from FedVLS) — KL on vacant-class logits
     between the frozen global model and the local model.
  3. Rare-class aggregation weighting — upweight clients with more rare-
     class pixels during server-side model averaging.

Local loss:
  L = L_focal_dice + alpha_p * L_proto + alpha_d * L_vacant_dis

Usage:
  python train_fedseis.py --num_clients 20 --split noniid --sample_ratio 0.25
  python train_fedseis.py --num_clients 20 --split noniid --sample_ratio 0.25 \
      --alpha_proto 0.5 --alpha_dis 0.1
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

NUM_ROUNDS    = 20
LOCAL_EPOCHS  = 3
BATCH_SIZE    = 4
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
NUM_CLASSES   = 6
DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED          = 42
RARE_CLASSES  = [4, 5]

# UNet decoder layers & their channel dims. Earlier layers = more abstract.
FEATURE_DIMS = {"up1": 512, "up2": 256, "up3": 128, "up4": 64}


# ── Feature-hook helper ─────────────────────────────────────────────────────
class FeatureExtractor:
    """Registers a forward hook on the chosen UNet decoder layer."""
    def __init__(self, model, layer_name):
        self.features = None
        module = getattr(model, layer_name)
        self.hook = module.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()


# ── Losses ──────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=1.0, dice_weight=1.0, gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma)
        self.dice = DiceLoss()
        self.fw = focal_weight
        self.dw = dice_weight

    def forward(self, logits, targets):
        return self.fw * self.focal(logits, targets) + self.dw * self.dice(logits, targets)


def prototype_alignment_loss(features, targets, prototypes, valid_mask):
    """
    Pull pixel features toward their class prototype.
      features:   (B, D, Hf, Wf)   — Hf/Wf may be < H/W for earlier layers
      targets:    (B, H, W) long
      prototypes: (C, D)
      valid_mask: (C,) bool
    Labels are nearest-downsampled to feature resolution so each feature
    pixel gets its majority class. Cosine-distance aligns normalized features
    to the class prototype.
    """
    B, D, Hf, Wf = features.shape
    _, H, W = targets.shape

    if (Hf, Wf) != (H, W):
        # Nearest downsample labels to feature resolution
        targets_ds = F.interpolate(
            targets.unsqueeze(1).float(), size=(Hf, Wf), mode="nearest"
        ).squeeze(1).long()
    else:
        targets_ds = targets

    feats_flat = features.permute(0, 2, 3, 1).reshape(-1, D)
    targets_flat = targets_ds.reshape(-1)

    feats_n = F.normalize(feats_flat, dim=1)
    protos_n = F.normalize(prototypes, dim=1)

    per_pixel_valid = valid_mask[targets_flat]
    if per_pixel_valid.sum() == 0:
        return torch.tensor(0.0, device=features.device)

    feats_v = feats_n[per_pixel_valid]
    protos_v = protos_n[targets_flat[per_pixel_valid]]

    cos = (feats_v * protos_v).sum(dim=1)
    return (1.0 - cos).mean()


def vacant_class_distillation(logits_local, logits_global, vacant_mask):
    if not vacant_mask.any():
        return torch.tensor(0.0, device=logits_local.device)
    vc_idx = vacant_mask.nonzero(as_tuple=True)[0]

    # Softmax over the FULL class dim, then select. Slicing before the softmax
    # makes the |vacant|==1 case degenerate: softmax over a length-1 dim is
    # identically 1.0, so the KL term is structurally zero.
    local_log_probs_full = F.log_softmax(logits_local, dim=1)
    global_probs_full = F.softmax(logits_global, dim=1)

    kl_full = global_probs_full * (
        torch.log(global_probs_full + 1e-10) - local_log_probs_full
    )
    return kl_full[:, vc_idx, :, :].sum(dim=1).mean()


# ── Partitioning & loading ──────────────────────────────────────────────────
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


def load_and_normalize(path):
    arr = np.load(path)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def build_client_loaders(train_seismic, train_labels, partitions, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    loaders = []
    for c, idxs in enumerate(partitions):
        ds = InlineLoader(
            seismic_cube=train_seismic, label_cube=train_labels,
            inline_inds=idxs, train_status=True, transform=transform,
        )
        loader = data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                 num_workers=2, pin_memory=True)
        loaders.append(loader)
        tag = str(idxs) if len(idxs) <= 10 else f"[{idxs[0]}..{idxs[-1]}]"
        print(f"  Client {c}: {len(idxs)} crosslines {tag}")
    return loaders


def build_test_loader(seismic_path, labels_path, batch_size=1):
    seismic = load_and_normalize(seismic_path)
    labels = np.load(labels_path)
    n = seismic.shape[1]
    transform = transforms.Compose([transforms.ToTensor()])
    ds = InlineLoader(
        seismic_cube=seismic, label_cube=labels,
        inline_inds=list(range(n)),
        train_status=False, transform=transform,
    )
    loader = data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    return loader, labels


def select_prototype_crosslines(train_labels, partitions, rng):
    """
    Pick ~1 crossline per class for the server's public prototype set.
    We prefer crosslines NOT assigned to any client (but in our geographic
    partition they all are, so we just pick crosslines with the richest
    presence of each class — this simulates a 'reference section' with
    full facies picks, standard assumption in seismic interpretation).
    Returns list[int] of crossline indices (union across classes).
    """
    all_assigned = set()
    for part in partitions:
        all_assigned.update(part)
    n_crosslines = train_labels.shape[1]
    chosen = set()
    for c in range(NUM_CLASSES):
        # Fraction of pixels that are class c per crossline
        mask = (train_labels == c)  # (inline, crossline, depth)
        per_cross = mask.sum(axis=(0, 2)) / (mask.shape[0] * mask.shape[2])
        # Pick crossline with highest fraction of class c
        best = int(np.argmax(per_cross))
        chosen.add(best)
    return sorted(chosen)


def build_prototype_loader(train_seismic, train_labels, proto_idxs, batch_size=1):
    transform = transforms.Compose([transforms.ToTensor()])
    ds = InlineLoader(
        seismic_cube=train_seismic, label_cube=train_labels,
        inline_inds=proto_idxs, train_status=False, transform=transform,
    )
    return data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)


# ── Prototype computation ───────────────────────────────────────────────────
def compute_prototypes(model, proto_loader, device, layer_name, feature_dim):
    """
    Forward pass over server's prototype slices, extract features at the
    chosen decoder layer, nearest-downsample labels to feature resolution,
    and compute per-class mean feature vector.
    Returns (prototypes, valid_mask): (C, D) tensor and (C,) bool.
    """
    model.eval()
    model.to(device)
    extractor = FeatureExtractor(model, layer_name)

    sums = torch.zeros(NUM_CLASSES, feature_dim, device=device)
    counts = torch.zeros(NUM_CLASSES, device=device)

    with torch.no_grad():
        for images, targets, _ in proto_loader:
            images = images.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)
            _ = model(images)  # populates extractor.features
            feats = extractor.features  # (B, D, Hf, Wf)
            B, D, Hf, Wf = feats.shape
            _, H, W = targets.shape
            if (Hf, Wf) != (H, W):
                targets_ds = F.interpolate(
                    targets.unsqueeze(1).float(), size=(Hf, Wf), mode="nearest"
                ).squeeze(1).long()
            else:
                targets_ds = targets
            feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, D)
            targets_flat = targets_ds.reshape(-1)
            for c in range(NUM_CLASSES):
                mask = (targets_flat == c)
                if mask.sum() > 0:
                    sums[c] += feats_flat[mask].sum(dim=0)
                    counts[c] += mask.sum()

    extractor.remove()
    valid = counts > 0
    prototypes = torch.zeros(NUM_CLASSES, feature_dim, device=device)
    prototypes[valid] = sums[valid] / counts[valid].unsqueeze(1)
    return prototypes, valid


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
    per_class = jaccard_score(
        test_labels.flatten(), prediction.flatten(),
        labels=list(range(NUM_CLASSES)), average=None,
    )
    return per_class.mean(), per_class


# ── Rare-weighted aggregation ───────────────────────────────────────────────
def weighted_aggregate(state_dicts, weights):
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    avg = copy.deepcopy(state_dicts[0])
    for key in avg.keys():
        avg[key] = avg[key] * w[0]
        for i in range(1, len(state_dicts)):
            avg[key] = avg[key] + state_dicts[i][key] * w[i]
    return avg


def compute_client_info(train_labels, partitions):
    info = []
    for c, idxs in enumerate(partitions):
        client_labels = train_labels[:, idxs, :].flatten()
        total = client_labels.size
        rare_frac = sum((client_labels == rc).sum() for rc in RARE_CLASSES) / total
        unique = len(np.unique(client_labels))
        info.append({"num_classes": unique, "rare_fraction": float(rare_frac)})
    return info


# ── Local training ──────────────────────────────────────────────────────────
def local_train_fedseis(model, global_model, loader, device, local_epochs,
                        lr, weight_decay, prototypes, proto_valid,
                        vacant_mask, alpha_proto, alpha_dis, proto_layer):
    model.train()
    model.to(device)
    global_model.eval()
    global_model.to(device)

    vacant_t = torch.tensor(vacant_mask, dtype=torch.bool, device=device)
    focal_dice = FocalDiceLoss()

    extractor = FeatureExtractor(model, proto_layer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(local_epochs):
        for images, targets, _ in loader:
            images = images.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)

            logits, _ = model(images)
            features = extractor.features  # (B, D, H, W)

            l_sup = focal_dice(logits, targets)

            # Prototype alignment
            l_proto = prototype_alignment_loss(
                features, targets, prototypes, proto_valid
            ) if alpha_proto > 0 else torch.tensor(0.0, device=device)

            # Vacant-class distillation
            if alpha_dis > 0 and vacant_t.any():
                with torch.no_grad():
                    global_logits, _ = global_model(images)
                l_dis = vacant_class_distillation(logits, global_logits, vacant_t)
            else:
                l_dis = torch.tensor(0.0, device=device)

            loss = l_sup + alpha_proto * l_proto + alpha_dis * l_dis
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    extractor.remove()
    return copy.deepcopy(model.state_dict())


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="FedSeis — prototype-based FL")
    p.add_argument("--num_clients", type=int, default=20)
    p.add_argument("--split", type=str, default="noniid", choices=["iid", "noniid"])
    p.add_argument("--num_rounds", type=int, default=NUM_ROUNDS)
    p.add_argument("--local_epochs", type=int, default=LOCAL_EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--sample_ratio", type=float, default=0.25)
    p.add_argument("--alpha_proto", type=float, default=0.5,
                   help="Prototype alignment loss weight")
    p.add_argument("--alpha_dis", type=float, default=0.1,
                   help="Vacant-class distillation weight")
    p.add_argument("--use_rare_agg", action="store_true", default=True,
                   help="Use rare-class-weighted aggregation")
    p.add_argument("--no_rare_agg", dest="use_rare_agg", action="store_false")
    p.add_argument("--proto_layer", type=str, default="up4",
                   choices=list(FEATURE_DIMS.keys()),
                   help="UNet decoder layer for prototype feature extraction")
    args = p.parse_args()
    feature_dim = FEATURE_DIMS[args.proto_layer]

    sr_tag = f"_sr{args.sample_ratio}" if args.sample_ratio < 1.0 else ""
    ra_tag = "_rare" if args.use_rare_agg else ""
    pl_tag = f"_{args.proto_layer}" if args.proto_layer != "up4" else ""
    exp_name = (f"fedseis_{args.split}_{args.num_clients}c_{args.num_rounds}r"
                f"{sr_tag}_ap{args.alpha_proto}_ad{args.alpha_dis}{ra_tag}{pl_tag}")
    save_dir = os.path.join(PROJECT_ROOT, "results", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print(f"FedSeis — Parihaka | {args.split.upper()} | {args.num_clients} clients")
    print("=" * 70)
    print(f"Sample ratio       : {args.sample_ratio}")
    print(f"alpha_proto        : {args.alpha_proto}")
    print(f"alpha_dis          : {args.alpha_dis}")
    print(f"Rare-weighted agg  : {args.use_rare_agg}")
    print(f"Proto layer        : {args.proto_layer} (dim={feature_dim})")
    print(f"Save directory     : {save_dir}")
    print("=" * 70)

    wandb.init(
        project="FL-Seismic",
        group=f"fedseis_{args.split}",
        name=exp_name,
        config={
            "algorithm": "FedSeis",
            "split_mode": args.split,
            "num_clients": args.num_clients,
            "sample_ratio": args.sample_ratio,
            "num_rounds": args.num_rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "alpha_proto": args.alpha_proto,
            "alpha_dis": args.alpha_dis,
            "use_rare_agg": args.use_rare_agg,
            "proto_layer": args.proto_layer,
            "feature_dim": feature_dim,
            "dataset": "Parihaka",
            "seed": args.seed,
        },
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Loading training data...")
    train_seismic = load_and_normalize(TRAIN_SEISMIC)
    train_labels = np.load(TRAIN_LABELS)
    num_crosslines = train_seismic.shape[1]
    print(f"  Training cube: {train_seismic.shape}")

    if args.split == "noniid":
        partitions = partition_noniid(num_crosslines, args.num_clients)
    else:
        partitions = partition_iid(num_crosslines, args.num_clients, rng)

    client_info = compute_client_info(train_labels, partitions)
    print("\nClient class info:")
    for c, info in enumerate(client_info):
        print(f"  Client {c}: {info['num_classes']} classes, rare_frac={info['rare_fraction']:.4f}")

    # Per-client vacant masks
    client_vacant = []
    for idxs in partitions:
        labels_c = train_labels[:, idxs, :].flatten()
        present = set(np.unique(labels_c).tolist())
        vacant = np.array([c not in present for c in range(NUM_CLASSES)], dtype=bool)
        client_vacant.append(vacant)

    # Select server prototype crosslines (public reference section)
    proto_idxs = select_prototype_crosslines(train_labels, partitions, rng)
    print(f"\nServer prototype crosslines: {proto_idxs} "
          f"({len(proto_idxs)}/{num_crosslines} = "
          f"{100*len(proto_idxs)/num_crosslines:.1f}%)")
    proto_seismic = train_seismic
    proto_loader = build_prototype_loader(proto_seismic, train_labels, proto_idxs)

    client_loaders = build_client_loaders(train_seismic, train_labels,
                                          partitions, args.batch_size)

    print("\nLoading test sets...")
    test_loader1, test_labels1 = build_test_loader(TEST1_SEISMIC, TEST1_LABELS)
    test_loader2, test_labels2 = build_test_loader(TEST2_SEISMIC, TEST2_LABELS)

    print("\nInitializing global UNet...")
    global_model = UNet(in_channels=1, num_classes=NUM_CLASSES, bilinear=False)
    print(f"  Parameters: {sum(p.numel() for p in global_model.parameters()):,}")

    best_miou = 0.0
    history = []

    print("\n" + "=" * 70)
    print(f"Starting FedSeis ({args.split.upper()}, {args.num_clients} clients)")
    print("=" * 70 + "\n")

    for round_idx in range(args.num_rounds):
        t0 = time.time()
        print(f"--- Round {round_idx + 1}/{args.num_rounds} ---")

        # ── Server: compute prototypes from its reference section ────────
        prototypes, proto_valid = compute_prototypes(
            global_model, proto_loader, DEVICE, args.proto_layer, feature_dim
        )
        valid_list = proto_valid.cpu().numpy().tolist()
        print(f"  Prototypes valid: {valid_list}")

        # ── Client subsampling ───────────────────────────────────────────
        if args.sample_ratio < 1.0:
            n_sampled = max(1, int(args.num_clients * args.sample_ratio))
            selected = sorted(rng.choice(args.num_clients, n_sampled, replace=False).tolist())
            print(f"  Sampled {n_sampled}/{args.num_clients} clients: {selected}")
        else:
            selected = list(range(args.num_clients))

        # ── Local training ──────────────────────────────────────────────
        global_weights = copy.deepcopy(global_model.state_dict())
        global_frozen = UNet(in_channels=1, num_classes=NUM_CLASSES, bilinear=False)
        global_frozen.load_state_dict(global_weights)
        global_frozen.to(DEVICE)

        client_weights = []
        for c in selected:
            local_model = UNet(in_channels=1, num_classes=NUM_CLASSES, bilinear=False)
            local_model.load_state_dict(global_weights)
            updated = local_train_fedseis(
                model=local_model, global_model=global_frozen,
                loader=client_loaders[c], device=DEVICE,
                local_epochs=args.local_epochs, lr=args.lr,
                weight_decay=WEIGHT_DECAY,
                prototypes=prototypes.detach(), proto_valid=proto_valid,
                vacant_mask=client_vacant[c],
                alpha_proto=args.alpha_proto, alpha_dis=args.alpha_dis,
                proto_layer=args.proto_layer,
            )
            client_weights.append(updated)
            print(f"  Client {c} done.")

        # ── Aggregation ─────────────────────────────────────────────────
        if args.use_rare_agg:
            raw = [max(client_info[c]["rare_fraction"], 0.01) for c in selected]
            w_norm = np.array(raw) / np.sum(raw)
            print(f"  Agg weights (rare): {['%.3f' % w for w in w_norm]}")
            aggregated = weighted_aggregate(client_weights, raw)
        else:
            aggregated = weighted_aggregate(client_weights, [1.0] * len(selected))
        global_model.load_state_dict(aggregated)

        # ── Eval ────────────────────────────────────────────────────────
        miou1, ci1 = evaluate_global(global_model, test_loader1, test_labels1, DEVICE)
        miou2, ci2 = evaluate_global(global_model, test_loader2, test_labels2, DEVICE)
        avg_miou = (miou1 + miou2) / 2.0
        avg_ci = (ci1 + ci2) / 2.0
        dt = time.time() - t0
        print(f"  T1 mIoU {miou1:.4f}  T2 mIoU {miou2:.4f}  Avg {avg_miou:.4f}")
        print(f"  Per-class IoU: {['%.4f' % v for v in avg_ci]}")
        print(f"  Time: {dt:.1f}s")

        log = {
            "round": round_idx + 1,
            "global/miou_test1": miou1,
            "global/miou_test2": miou2,
            "global/miou_avg": avg_miou,
            "global/round_time_s": dt,
        }
        for i in range(NUM_CLASSES):
            log[f"per_class_iou_avg/class_{i}"] = avg_ci[i]
            log[f"prototype_valid/class_{i}"] = int(valid_list[i])
        wandb.log(log, step=round_idx + 1)

        if avg_miou > best_miou:
            best_miou = avg_miou
            torch.save(global_model.state_dict(),
                       os.path.join(save_dir, "best_global_model.pth"))
            print(f"  ** New best (mIoU={best_miou:.4f}) **")

        torch.save(global_model.state_dict(),
                   os.path.join(save_dir, f"global_round_{round_idx}.pth"))

        history.append({
            "round": round_idx + 1,
            "miou_test1": miou1, "miou_test2": miou2, "avg_miou": avg_miou,
            "per_class_iou_test1": ci1.tolist(),
            "per_class_iou_test2": ci2.tolist(),
            "time_s": dt,
        })
        print()

    print("=" * 70)
    print(f"Training complete. Best mIoU: {best_miou:.4f}")
    print("=" * 70)

    np.save(os.path.join(save_dir, "training_history.npy"), history, allow_pickle=True)
    wandb.run.summary["best_avg_miou"] = best_miou
    art = wandb.Artifact(f"best_model_{exp_name}", type="model")
    art.add_file(os.path.join(save_dir, "best_global_model.pth"))
    wandb.log_artifact(art)
    wandb.finish()


if __name__ == "__main__":
    main()
