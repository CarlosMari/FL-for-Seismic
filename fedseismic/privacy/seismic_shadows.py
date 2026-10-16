"""12-shadow pilot: calibrate the seismic count line on one config.

N=5 FedAvg batchnorm, seed 0. Each shadow is one 5-epoch UNet update from
the round-20 received model on a random-boundary contiguous chunk (same
geometry as the victims, varied counts), with the victim's exact optimizer
settings. The count line (row/bias size to pixel counts) is fit on the 12
shadows and applied to the 5 victims.

Decision rule (fixed before running): if the shadow fit separates clearly
above chance, report the victim Pearson as calibrated with 12-shadow
intervals. If near chance, report as a negative calibration result. No
expansion either way. F3 only, single seed.
"""

from pathlib import Path

import numpy as np
import torch

from fedseismic.config import RunConfig, resolve_device
from fedseismic.experiment import build_federated_run
from fedseismic.data.seismic import _to_tensor
from fedseismic.data.partition import build_client_loaders
from fedseismic.federated.client import ClientTrainer
from fedseismic.privacy.estimators import extract_last_linear
from fedseismic.privacy.log import _load_checkpoint, load_labels, load_meta
from fedseismic.privacy.metrics import to_simplex

N_SHADOWS = 12
N_SITES = 5
SEED = 0
JOB = "fedavg_noniid_a0.1_lNA_dna_n5_batchnorm"


def _random_chunks(total, parts, rng):
    """Random contiguous partition of [0, total) into `parts` chunks."""
    cuts = sorted(rng.choice(total - 1, size=parts - 1, replace=False) + 1)
    bounds = [0] + [int(c) for c in cuts] + [total]
    return [list(range(bounds[i], bounds[i + 1])) for i in range(parts)]


def _pixel_counts(train_labels, part, num_classes):
    labels = np.asarray(train_labels)[:, part, :].reshape(-1)
    return np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64)


def run_pilot(n_shadows=N_SHADOWS, device=None):
    from fedseismic.data.seismic import load_and_normalize

    device = resolve_device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run = Path("results/runs/seismic_f3") / JOB / f"seed_{SEED}"
    meta = load_meta(run)
    cfg = RunConfig.from_mapping(meta["config"])
    seed = int(meta.get("seed", cfg.seed))
    data = build_federated_run(cfg, seed)
    start = _load_checkpoint(run / "received" / "round_020.pt")
    train_seismic = load_and_normalize(cfg.train_seismic)
    train_labels = np.load(cfg.train_labels)

    rng = np.random.RandomState(90_000 + seed)
    from fedseismic.experiment import _criterion
    criterion = _criterion(cfg, data.train_labels)
    features, counts = [], []
    for index in range(n_shadows):
        chunks = _random_chunks(train_seismic.shape[1], N_SITES, rng)
        pick = int(rng.randint(N_SITES))
        part = chunks[pick]
        loader = build_client_loaders(
            train_seismic, train_labels, [part], cfg.batch_size,
            num_workers=0, pin_memory=str(device).startswith("cuda"),
        )[0]
        trainer = ClientTrainer(
            data.model_factory(), loader, criterion, device=device,
            local_epochs=cfg.local_epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
            optimizer=cfg.optimizer, momentum=cfg.momentum,
            num_classes=cfg.num_classes,
        )
        trainer.download(start)
        torch.manual_seed(95_000 + seed + index)
        trainer.train(global_state=start)
        uploaded = trainer.upload()
        local_weight, local_bias = extract_last_linear(uploaded)
        received_weight, received_bias = extract_last_linear(start)
        delta_w = np.asarray(local_weight, dtype=np.float64) - np.asarray(received_weight, dtype=np.float64)
        delta_b = np.asarray(local_bias, dtype=np.float64).reshape(-1) - np.asarray(received_bias, dtype=np.float64).reshape(-1)
        features.append((np.linalg.norm(delta_w.reshape(delta_w.shape[0], -1), axis=1), delta_b))
        counts.append(_pixel_counts(train_labels, part, cfg.num_classes))
        print(f"  shadow {index + 1}/{n_shadows}", flush=True)
    out = Path("results/attacks/seismic_shadows") / JOB / f"seed_{SEED}" / "shadow.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        row=np.stack([f[0] for f in features]),
        bias=np.stack([f[1] for f in features]),
        counts=np.stack(counts),
    )
    print(f"wrote {out.resolve()}")
    return out


if __name__ == "__main__":
    run_pilot()
