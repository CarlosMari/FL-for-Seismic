"""
Visualize segmentation predictions vs ground truth for a saved FL model.

Usage:
    python visualize_predictions.py --model_path results/fedavg_noniid_20c_20r_sr0.25/best_global_model.pth
    python visualize_predictions.py --model_path results/fedavg_noniid_20c_20r_sr0.25/best_global_model.pth --num_slices 5
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from train_tools.models.unet import UNet
from train_tools.preprocessing.seismic.datasets import InlineLoader
from torchvision.transforms import transforms
import torch.utils.data as data

NUM_CLASSES = 6
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Class colormap — distinct colors for 6 facies
FACIES_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
FACIES_CMAP = ListedColormap(FACIES_COLORS)
FACIES_NORM = BoundaryNorm(np.arange(-0.5, NUM_CLASSES, 1), NUM_CLASSES)
FACIES_NAMES = [f'Class {i}' for i in range(NUM_CLASSES)]


def load_and_normalize(path):
    arr = np.load(path)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def predict_cube(model, seismic, labels, crossline_idxs, device):
    """Run inference and return prediction cube matching labels shape."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = InlineLoader(
        seismic_cube=seismic, label_cube=labels,
        inline_inds=crossline_idxs, train_status=False, transform=transform,
    )
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    prediction = np.zeros(labels.shape)
    model.eval()
    model.to(device)
    sample_idx = 0
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device, dtype=torch.float)
            logits, _ = model(images)
            preds = logits.argmax(dim=1)
            for b in range(images.size(0)):
                prediction[:, sample_idx, :] = preds[b].cpu().numpy().T
                sample_idx += 1
    return prediction


def plot_slices(seismic, labels, prediction, crossline_idxs, save_dir, prefix="test"):
    """Plot seismic input, ground truth, prediction, and error map for selected crosslines."""
    os.makedirs(save_dir, exist_ok=True)

    for idx in crossline_idxs:
        seis_slice = seismic[:, idx, :]    # (inline, depth)
        gt_slice = labels[:, idx, :]       # (inline, depth)
        pred_slice = prediction[:, idx, :] # (inline, depth)
        error_slice = (gt_slice != pred_slice).astype(float)

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        # Seismic
        axes[0].imshow(seis_slice.T, cmap='gray', aspect='auto')
        axes[0].set_title(f'Seismic (crossline {idx})')
        axes[0].set_ylabel('Depth')
        axes[0].set_xlabel('Inline')

        # Ground truth
        im_gt = axes[1].imshow(gt_slice.T, cmap=FACIES_CMAP, norm=FACIES_NORM, aspect='auto')
        axes[1].set_title('Ground Truth')
        axes[1].set_xlabel('Inline')

        # Prediction
        axes[2].imshow(pred_slice.T, cmap=FACIES_CMAP, norm=FACIES_NORM, aspect='auto')
        axes[2].set_title('Prediction')
        axes[2].set_xlabel('Inline')

        # Error map (colored by GT class where wrong)
        error_colored = np.full_like(gt_slice, -1, dtype=float)
        error_colored[gt_slice != pred_slice] = gt_slice[gt_slice != pred_slice]
        error_cmap = ListedColormap(['white'] + FACIES_COLORS)
        error_norm = BoundaryNorm(np.arange(-1.5, NUM_CLASSES, 1), NUM_CLASSES + 1)
        axes[3].imshow(error_colored.T, cmap=error_cmap, norm=error_norm, aspect='auto')
        axes[3].set_title('Errors (colored by GT class)')
        axes[3].set_xlabel('Inline')

        # Colorbar
        cbar = fig.colorbar(im_gt, ax=axes, orientation='horizontal', fraction=0.03, pad=0.08,
                           ticks=range(NUM_CLASSES))
        cbar.ax.set_xticklabels(FACIES_NAMES)

        plt.tight_layout()
        path = os.path.join(save_dir, f'{prefix}_crossline_{idx}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize FL segmentation predictions")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_slices", type=int, default=5,
                        help="Number of evenly-spaced crosslines to visualize per test set")
    parser.add_argument("--dataset", type=str, default="parihaka", choices=["parihaka", "f3"])
    args = parser.parse_args()

    model_dir = os.path.dirname(args.model_path)
    save_dir = os.path.join(model_dir, "visualizations")

    print(f"Loading model from: {args.model_path}")
    model = UNet(in_channels=1, num_classes=NUM_CLASSES, bilinear=False)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    print(f"  Device: {DEVICE}")

    if args.dataset == "parihaka":
        # Test set 1
        print("\nProcessing Parihaka Test1...")
        test1_seis = load_and_normalize(os.path.join(PROJECT_ROOT, "datasets", "test_once", "test1_seismic.npy"))
        test1_labels = np.load(os.path.join(PROJECT_ROOT, "datasets", "test_once", "test1_labels.npy"))
        n_cross1 = test1_seis.shape[1]
        pred1 = predict_cube(model, test1_seis, test1_labels, list(range(n_cross1)), DEVICE)
        idxs1 = np.linspace(0, n_cross1 - 1, args.num_slices, dtype=int)
        print(f"  Visualizing crosslines: {idxs1.tolist()}")
        plot_slices(test1_seis, test1_labels, pred1, idxs1, save_dir, prefix="test1")

        # Test set 2
        print("\nProcessing Parihaka Test2...")
        test2_seis = load_and_normalize(os.path.join(PROJECT_ROOT, "datasets", "test_once", "test2_seismic.npy"))
        test2_labels = np.load(os.path.join(PROJECT_ROOT, "datasets", "test_once", "test2_labels.npy"))
        n_cross2 = test2_seis.shape[1]
        pred2 = predict_cube(model, test2_seis, test2_labels, list(range(n_cross2)), DEVICE)
        idxs2 = np.linspace(0, n_cross2 - 1, args.num_slices, dtype=int)
        print(f"  Visualizing crosslines: {idxs2.tolist()}")
        plot_slices(test2_seis, test2_labels, pred2, idxs2, save_dir, prefix="test2")

    else:  # f3
        import segyio
        F3_SEGY = os.path.join(PROJECT_ROOT, "..", "OCT_SEISMIC_PIPELINE", "data", "f3", "f3_train.segy")
        F3_LABELS = os.path.join(PROJECT_ROOT, "..", "OCT_SEISMIC_PIPELINE", "data", "f3", "f3_train_labels.npy")
        print("\nLoading F3 data...")
        with segyio.open(F3_SEGY, ignore_geometry=True) as f:
            traces = np.array([f.trace[i] for i in range(f.tracecount)])
        seismic = traces.reshape(401, 701, 255)
        seismic = (seismic - seismic.min()) / (seismic.max() - seismic.min() + 1e-8)
        labels = np.load(F3_LABELS)
        n_train = 701 - int(701 * 0.2)
        test_crosslines = list(range(n_train, 701))
        test_labels = labels[:, test_crosslines, :]
        pred = predict_cube(model, seismic, labels, test_crosslines, DEVICE)
        # pred shape matches test_labels
        # But for plotting we need to index into the test subset
        n_test = len(test_crosslines)
        idxs = np.linspace(0, n_test - 1, args.num_slices, dtype=int)
        print(f"  Visualizing test crosslines: {[test_crosslines[i] for i in idxs]}")
        # For plotting, use the test subset directly
        test_seis = seismic[:, test_crosslines, :]
        plot_slices(test_seis, test_labels, pred, idxs, save_dir, prefix="f3_test")

    print(f"\nAll visualizations saved to: {save_dir}")


if __name__ == "__main__":
    main()
