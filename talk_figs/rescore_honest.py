"""Honest re-score: all-class + buffered (drop K test crosslines next to the
train boundary) for every checkpoint. Inference-only. Quantifies the two known
inflations (present-class, adjacency) so the thesis can put NUMBERS on them."""
import glob
import numpy as np
import torch
from sklearn.metrics import jaccard_score
from fedseismic.config import RunConfig
from fedseismic.experiment import build_federated_run
from fedseismic.models import UNet
from fedseismic.privacy.log import load_meta, load_personal_states, load_local_states, _load_checkpoint
from fedseismic.eval.metrics import IGNORE


def score(model, loader, num_classes, block, K=0):
    model.eval()
    preds, targs = [], []
    n = 0
    with torch.no_grad():
        for img, tgt, _ in loader:
            if n < K:
                n += 1
                continue
            lg = model(img.float())[0] if isinstance(model(img.float()), (tuple, list)) else model(img.float())
            lg = lg[:, block[0]:block[1]].clone()
            preds.append(lg.argmax(1).cpu().numpy().ravel() + block[0])
            targs.append(np.asarray(tgt).ravel())
            n += 1
    pr = np.concatenate(preds); gt = np.concatenate(targs)
    keep = gt != IGNORE
    pr, gt = pr[keep], gt[keep]
    per = jaccard_score(gt, pr, labels=list(range(num_classes)), average=None, zero_division=0)
    pres = np.array([np.any(gt == k) for k in range(num_classes)])
    present = float(per[block[0]:block[1]][pres[block[0]:block[1]]].mean())
    allc = float(per[block[0]:block[1]].mean())
    return present, allc


def run(run_dir, site, block, ks=(0, 10, 25, 50), device="cpu"):
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    data = build_federated_run(cfg, int(meta.get("seed", 0)))
    try:
        st = load_personal_states(run_dir)[site]
    except Exception:
        st = load_local_states(run_dir)[site]
    m = UNet(1, cfg.num_classes, bilinear=False, norm=cfg.norm, groups=cfg.norm_groups)
    m.load_state_dict(st)
    print(f"== {run_dir}  site {site} ==")
    for K in ks:
        p, a = score(m, data.client_tests[site], cfg.num_classes, block, K=K)
        print(f"  buffer K={K:2d}: present={p:.4f}  all-class={a:.4f}")


if __name__ == "__main__":
    import sys
    run(sys.argv[1], int(sys.argv[2]), (int(sys.argv[3]), int(sys.argv[4])))
