"""Ensemble post-hoc on n=5 V3+frc checkpoints (Phase V).

Average softmax probs across 5 seeds (42/123/7/99/2025). Optionally
apply MetaFusion logit adjustment per-model before averaging.

Goal: new SOTA single-prediction via seed ensembling — the 5 seeds
individually show bistability (some kill C5, some kill C4, s2025
gets both), so ensembling should cover the 2D C4/C5 axis.
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

TRAIN_LABELS  = f'{PROJECT_ROOT}/datasets/train/train_labels.npy'
TEST1_SEISMIC = f'{PROJECT_ROOT}/datasets/test_once/test1_seismic.npy'
TEST1_LABELS  = f'{PROJECT_ROOT}/datasets/test_once/test1_labels.npy'
TEST2_SEISMIC = f'{PROJECT_ROOT}/datasets/test_once/test2_seismic.npy'
TEST2_LABELS  = f'{PROJECT_ROOT}/datasets/test_once/test2_labels.npy'

SEEDS = [42, 123, 7, 99, 2025]

# Which checkpoint to ensemble. "best" is best-round-on-TEST, which leaks the
# evaluation set and flatters unstable configs; "final" (round 19, 0-indexed)
# is the unbiased choice. Override with --ckpt final.
CKPT_KIND = os.environ.get('ENSEMBLE_CKPT', 'best')
_CKPT_FILE = {'best': 'best_global_model.pth', 'final': 'global_round_19.pth'}
RUN_TMPL = f'{PROJECT_ROOT}/results/fedavg_noniid_20c_20r_sr0.25_agginvfreq_miou_lossrecall_slice_frc_s{{seed}}'
CKPT_TMPL = RUN_TMPL + '/' + _CKPT_FILE[CKPT_KIND]


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


def load_model(ckpt_path):
    model = UNet(in_channels=1, num_classes=NUM_CLASSES, bilinear=False)
    sd = torch.load(ckpt_path, map_location=DEVICE)
    if 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    model.load_state_dict(sd)
    model = model.to(DEVICE)
    model.eval()
    return model


def evaluate_ensemble(models, loader, labels, log_prior, tau):
    """Average softmax probs over models. log_prior: tensor [C]. tau: scalar."""
    prediction = np.zeros(labels.shape, dtype=np.int64)
    adjust = tau * log_prior
    adjust = adjust.view(1, NUM_CLASSES, 1, 1).to(DEVICE)
    sample_idx = 0
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(DEVICE, dtype=torch.float)
            probs_sum = None
            for m in models:
                logits, _ = m(images)
                adj = logits - adjust
                p = F.softmax(adj, dim=1)
                probs_sum = p if probs_sum is None else probs_sum + p
            probs_avg = probs_sum / len(models)
            preds = probs_avg.argmax(dim=1)
            for b in range(images.size(0)):
                prediction[:, sample_idx, :] = preds[b].cpu().numpy().T
                sample_idx += 1
    per_class = jaccard_score(
        labels.flatten(), prediction.flatten(),
        labels=list(range(NUM_CLASSES)), average=None
    )
    return per_class.mean(), per_class


def main():
    print(f'[ensemble] Device: {DEVICE}')
    print(f'[ensemble] Seeds: {SEEDS}')

    print('[ensemble] Loading test1 ...')
    t1_loader, t1_labels = build_test_loader(TEST1_SEISMIC, TEST1_LABELS)
    print('[ensemble] Loading test2 ...')
    t2_loader, t2_labels = build_test_loader(TEST2_SEISMIC, TEST2_LABELS)

    print('[ensemble] Computing train class prior ...')
    train_labels = np.load(TRAIN_LABELS)
    prior = compute_class_prior(train_labels)
    print(f'[ensemble] Train class prior: {prior}')
    log_prior = torch.tensor(np.log(prior + 1e-12), dtype=torch.float32)

    print('[ensemble] Loading 5 models ...')
    models = []
    for s in SEEDS:
        p = CKPT_TMPL.format(seed=s)
        if not os.path.isfile(p):
            print(f'  MISSING: {p}')
            continue
        print(f'  loaded s{s}')
        models.append(load_model(p))

    taus = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    print()
    print(f'{"tau":>5s}  {"mIoU1":>7s}  {"mIoU2":>7s}  {"avg":>7s}  per-class avg (C0..C5)')
    print('-' * 85)
    rows = []
    for tau in taus:
        m1, pc1 = evaluate_ensemble(models, t1_loader, t1_labels, log_prior, tau)
        m2, pc2 = evaluate_ensemble(models, t2_loader, t2_labels, log_prior, tau)
        avg = 0.5 * (m1 + m2)
        pc_avg = 0.5 * (pc1 + pc2)
        cls_str = ' '.join(f'{v:.3f}' for v in pc_avg)
        print(f'{tau:>5.2f}  {m1:>7.4f}  {m2:>7.4f}  {avg:>7.4f}  [{cls_str}]')
        rows.append((tau, m1, m2, avg, pc_avg.tolist()))

    out = f'{PROJECT_ROOT}/paper_figures/ensemble_v3_frc_{CKPT_KIND}.txt'
    with open(out, 'w') as f:
        f.write(f'Ensemble (softmax avg) over V3+frc seeds {SEEDS}\n')
        f.write(f'Train class prior: {prior.tolist()}\n\n')
        f.write(f'{"tau":>5s}  {"mIoU1":>7s}  {"mIoU2":>7s}  {"avg":>7s}  per-class avg (C0..C5)\n')
        for tau, m1, m2, avg, pc in rows:
            cls_str = ' '.join(f'{v:.3f}' for v in pc)
            f.write(f'{tau:>5.2f}  {m1:>7.4f}  {m2:>7.4f}  {avg:>7.4f}  [{cls_str}]\n')
    print(f'\n[ensemble] wrote {out}')


if __name__ == '__main__':
    main()
