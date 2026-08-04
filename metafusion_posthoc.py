"""Post-hoc MetaFusion-style logit adjustment on P3 s7 checkpoint.

MetaFusion (Chan et al 2019) / logit adjustment (Menon et al 2021):
at inference, shift logits by -tau * log(class_prior). This lifts
rare-class probability at no training cost.

We scan tau in {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0} and report
per-class IoU and mean mIoU on test1/test2 for the P3 s7 best checkpoint
(mIoU 0.5932, finalC5 0.280).
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils import data as tdata
from sklearn.metrics import jaccard_score

PROJECT_ROOT = '/home/carlos/projects/FL-for-Seismic'
sys.path.insert(0, PROJECT_ROOT)

from train_tools.models.unet import UNet
from train_tools.preprocessing.seismic.datasets import InlineLoader

NUM_CLASSES = 6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_SEISMIC = f'{PROJECT_ROOT}/datasets/train/train_seismic.npy'
TRAIN_LABELS  = f'{PROJECT_ROOT}/datasets/train/train_labels.npy'
TEST1_SEISMIC = f'{PROJECT_ROOT}/datasets/test_once/test1_seismic.npy'
TEST1_LABELS  = f'{PROJECT_ROOT}/datasets/test_once/test1_labels.npy'
TEST2_SEISMIC = f'{PROJECT_ROOT}/datasets/test_once/test2_seismic.npy'
TEST2_LABELS  = f'{PROJECT_ROOT}/datasets/test_once/test2_labels.npy'

CHECKPOINT = os.environ.get(
    'METAFUSION_CKPT',
    f'{PROJECT_ROOT}/results/fedavg_noniid_20c_20r_sr0.25_aggrare_lossrecall_slice/best_global_model.pth'
)


def load_and_normalize(path):
    arr = np.load(path)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def build_test_loader(seismic_path, labels_path):
    seismic = load_and_normalize(seismic_path)
    labels = np.load(labels_path)
    num_crosslines = seismic.shape[1]
    ds = InlineLoader(
        seismic_cube=seismic, label_cube=labels,
        inline_inds=list(range(num_crosslines)),
        train_status=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    loader = tdata.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    return loader, labels


def compute_class_prior(labels):
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for c in range(NUM_CLASSES):
        counts[c] = (labels == c).sum()
    return counts / counts.sum()


def evaluate_with_tau(model, loader, labels, log_prior, tau):
    """log_prior: torch tensor shape [C]. tau: scalar."""
    model.eval()
    prediction = np.zeros(labels.shape, dtype=np.int64)
    adjust = tau * log_prior  # [C]
    adjust = adjust.view(1, NUM_CLASSES, 1, 1).to(DEVICE)
    sample_idx = 0
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(DEVICE, dtype=torch.float)
            logits, _ = model(images)
            adj_logits = logits - adjust
            preds = adj_logits.argmax(dim=1)
            for b in range(images.size(0)):
                prediction[:, sample_idx, :] = preds[b].cpu().numpy().T
                sample_idx += 1
    per_class = jaccard_score(
        labels.flatten(), prediction.flatten(),
        labels=list(range(NUM_CLASSES)), average=None
    )
    return per_class.mean(), per_class


def main():
    print(f'[metafusion] Device: {DEVICE}')
    print(f'[metafusion] Checkpoint: {CHECKPOINT}')

    # Build test loaders
    print('[metafusion] Loading test1 ...')
    t1_loader, t1_labels = build_test_loader(TEST1_SEISMIC, TEST1_LABELS)
    print('[metafusion] Loading test2 ...')
    t2_loader, t2_labels = build_test_loader(TEST2_SEISMIC, TEST2_LABELS)

    # Compute class prior from training labels
    print('[metafusion] Computing train class prior ...')
    train_labels = np.load(TRAIN_LABELS)
    prior = compute_class_prior(train_labels)
    print(f'[metafusion] Train class prior: {prior}')
    log_prior = torch.tensor(np.log(prior + 1e-12), dtype=torch.float32)

    # Load model
    print('[metafusion] Loading model ...')
    model = UNet(in_channels=1, num_classes=NUM_CLASSES, bilinear=False)
    sd = torch.load(CHECKPOINT, map_location=DEVICE)
    if 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    model.load_state_dict(sd)
    model = model.to(DEVICE)

    taus = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    print()
    print(f'{"tau":>5s}  {"mIoU1":>7s}  {"mIoU2":>7s}  {"avg":>7s}  per-class avg (C0..C5)')
    print('-' * 85)
    rows = []
    for tau in taus:
        m1, pc1 = evaluate_with_tau(model, t1_loader, t1_labels, log_prior, tau)
        m2, pc2 = evaluate_with_tau(model, t2_loader, t2_labels, log_prior, tau)
        avg = 0.5 * (m1 + m2)
        pc_avg = 0.5 * (pc1 + pc2)
        cls_str = ' '.join(f'{v:.3f}' for v in pc_avg)
        print(f'{tau:>5.2f}  {m1:>7.4f}  {m2:>7.4f}  {avg:>7.4f}  [{cls_str}]')
        rows.append((tau, m1, m2, avg, pc_avg.tolist()))

    # Save results
    out = f'{PROJECT_ROOT}/paper_figures/metafusion_results.txt'
    with open(out, 'w') as f:
        f.write('MetaFusion / logit-adjustment scan on P3 s7 (Recall-per-slice, best mIoU 0.5932)\n')
        f.write(f'Checkpoint: {CHECKPOINT}\n')
        f.write(f'Train class prior: {prior.tolist()}\n\n')
        f.write(f'{"tau":>5s}  {"mIoU1":>7s}  {"mIoU2":>7s}  {"avg":>7s}  per-class avg (C0..C5)\n')
        for tau, m1, m2, avg, pc in rows:
            cls_str = ' '.join(f'{v:.3f}' for v in pc)
            f.write(f'{tau:>5.2f}  {m1:>7.4f}  {m2:>7.4f}  {avg:>7.4f}  [{cls_str}]\n')
    print(f'\n[metafusion] wrote {out}')


if __name__ == '__main__':
    main()
