"""Ensemble comparison: V3+frc vs T vs V3+frc∪T (10 seeds) vs single-config.

Goal: check if ensemble gain is orthogonal to aggregation choice, and
whether cross-config ensembling wins.
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
V3_FRC_TMPL = f'{PROJECT_ROOT}/results/fedavg_noniid_20c_20r_sr0.25_agginvfreq_miou_lossrecall_slice_frc_s{{seed}}/best_global_model.pth'
T_TMPL      = f'{PROJECT_ROOT}/results/fedavg_noniid_20c_20r_sr0.25_agginvfreq_invmiou_lossrecall_slice_s{{seed}}/best_global_model.pth'


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


def run_config(tag, ckpt_paths, t1_loader, t1_labels, t2_loader, t2_labels, log_prior, taus, out_file):
    print(f'\n### ENSEMBLE CONFIG: {tag} ({len(ckpt_paths)} models) ###')
    models = []
    for p in ckpt_paths:
        if not os.path.isfile(p):
            print(f'  MISSING: {p}')
            continue
        print(f'  loaded {os.path.basename(os.path.dirname(p))}')
        models.append(load_model(p))

    print(f'{"tau":>5s}  {"mIoU1":>7s}  {"mIoU2":>7s}  {"avg":>7s}  per-class avg (C0..C5)')
    print('-' * 85)
    out_file.write(f'\n### {tag} ({len(models)} models) ###\n')
    out_file.write(f'{"tau":>5s}  {"mIoU1":>7s}  {"mIoU2":>7s}  {"avg":>7s}  per-class avg (C0..C5)\n')
    for tau in taus:
        m1, pc1 = evaluate_ensemble(models, t1_loader, t1_labels, log_prior, tau)
        m2, pc2 = evaluate_ensemble(models, t2_loader, t2_labels, log_prior, tau)
        avg = 0.5 * (m1 + m2)
        pc_avg = 0.5 * (pc1 + pc2)
        cls_str = ' '.join(f'{v:.3f}' for v in pc_avg)
        line = f'{tau:>5.2f}  {m1:>7.4f}  {m2:>7.4f}  {avg:>7.4f}  [{cls_str}]'
        print(line)
        out_file.write(line + '\n')
    out_file.flush()

    # free GPU mem
    for m in models:
        del m
    torch.cuda.empty_cache()


def main():
    print(f'[ensemble_compare] Device: {DEVICE}')

    print('[ensemble_compare] Loading test sets ...')
    t1_loader, t1_labels = build_test_loader(TEST1_SEISMIC, TEST1_LABELS)
    t2_loader, t2_labels = build_test_loader(TEST2_SEISMIC, TEST2_LABELS)

    train_labels = np.load(TRAIN_LABELS)
    prior = compute_class_prior(train_labels)
    log_prior = torch.tensor(np.log(prior + 1e-12), dtype=torch.float32)

    taus = [0.0, 0.5, 0.75]

    v3_paths = [V3_FRC_TMPL.format(seed=s) for s in SEEDS]
    t_paths  = [T_TMPL.format(seed=s) for s in SEEDS]
    all_paths = v3_paths + t_paths

    out_path = f'{PROJECT_ROOT}/paper_figures/ensemble_compare.txt'
    with open(out_path, 'w') as f:
        f.write(f'Ensemble comparison: V3+frc vs T vs V3+frc∪T\n')
        f.write(f'Seeds: {SEEDS}\n')
        run_config('V3+frc (5)',     v3_paths,  t1_loader, t1_labels, t2_loader, t2_labels, log_prior, taus, f)
        run_config('T boost-bad (5)', t_paths,  t1_loader, t1_labels, t2_loader, t2_labels, log_prior, taus, f)
        run_config('V3+frc ∪ T (10)', all_paths, t1_loader, t1_labels, t2_loader, t2_labels, log_prior, taus, f)

    print(f'\n[ensemble_compare] wrote {out_path}')


if __name__ == '__main__':
    main()
