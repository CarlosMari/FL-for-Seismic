"""Zero-GPU check: does restricting the 12-class head to each survey's own
6 classes close the gap to solo? If yes, the 12-class softmax head-conflict
is the mechanism, and shared-body + per-survey heads should recover it.

For each survey's deployed FedBN model (shared weights + own BN), predict on
its holdout with argmax restricted to that survey's class block (F3 = 0-5,
Parihaka = 6-11), and score present-class mIoU against the solo baseline.
"""

from pathlib import Path
import numpy as np
import torch


def restricted_miou(model, loader, num_classes, class_block, device):
    """Present-class mIoU with argmax restricted to ``class_block``."""
    from fedseismic.eval.metrics import _logits, _present_mean
    from sklearn.metrics import jaccard_score

    model.eval()
    model.to(device)
    lo, hi = class_block
    mask = torch.zeros(num_classes, dtype=torch.bool)
    mask[lo:hi] = True
    preds, targs = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            logits = _logits(model(images.to(device, dtype=torch.float)))
            logits = logits.clone()
            logits[:, ~mask.to(device)] = -1e9
            preds.append(logits.argmax(dim=1).cpu().numpy().ravel())
            targs.append(np.asarray(labels.detach().cpu() if torch.is_tensor(labels) else labels).ravel())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targs)
    per = jaccard_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)
    return _present_mean(per, y_true, num_classes), np.asarray(per)


def main(run, device="cpu"):
    from fedseismic.config import RunConfig
    from fedseismic.experiment import build_federated_run
    from fedseismic.models import UNet
    from fedseismic.privacy.log import load_local_states, load_meta

    run = Path(run)
    meta = load_meta(run)
    cfg = RunConfig.from_mapping(meta["config"])
    seed = int(meta.get("seed", cfg.seed))
    data = build_federated_run(cfg, seed)
    states = load_local_states(run)
    blocks = {"A": (0, 6), "B": (6, 12)}
    for survey, block in blocks.items():
        for client, state in sorted(states.items()):
            loader = data.client_tests[client]
            if loader is None or not len(loader.dataset):
                continue
            m = UNet(1, cfg.num_classes, bilinear=False, norm=cfg.norm, groups=cfg.norm_groups)
            m.load_state_dict(state)
            miou, per = restricted_miou(m, loader, cfg.num_classes, block, device)
            print(f"survey {survey} client {client} restricted-miou={miou:.4f} per-class={np.round(per,3)}", flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "results/runs/multisurvey_fedbn/fedbn_noniid_a0.1_lNA_dna_n4/seed_0")
