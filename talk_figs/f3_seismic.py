"""F3 (redesigned) - "The blackout", shown as SEISMIC + facies overlay.

Same Parihaka crossline in all panels, grayscale seismic behind the facies
overlay so an interpreter sees the section. (a) averaged BN -> the facies
overlay collapses to one facies (the blackout) and no longer tracks the
seismic; (b) local BN -> recovered, tracks the structure; (c) interpretation;
(d) FedBN. This reads instantly as "averaged BN loses the geology".
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import torch
from fedseismic.config import RunConfig
from fedseismic.experiment import build_federated_run
from fedseismic.models import UNet
from fedseismic.privacy.log import load_meta, load_global_state, load_local_states
from talk_figs import style as S

BLOCK = (6, 12)  # Parihaka = union rows 6-11
NAMES = S.PARIHAKA_NAMES


def predict_masked(model, img, device="cpu"):
    model.eval()
    with torch.no_grad():
        lg = model(img.unsqueeze(0).to(device, dtype=torch.float))
        lg = lg[0] if isinstance(lg, (tuple, list)) else lg
        lg = lg[:, BLOCK[0]:BLOCK[1]].clone()
        return lg.argmax(1).squeeze(0).cpu().numpy() + BLOCK[0]


def reestimate_bn(model, train_loader, device="cpu"):
    model.train()
    with torch.no_grad():
        for img, _, _ in train_loader:
            model(img.to(device, dtype=torch.float))
    model.eval()
    return model


def main():
    run = "results/runs/multisurvey/fedavg_noniid_a0.1_lNA_dna_n4/seed_0"
    fedbn_run = "results/runs/multisurvey_fedbn/fedbn_noniid_a0.1_lNA_dna_n4/seed_0"
    meta = load_meta(run)
    cfg = RunConfig.from_mapping(meta["config"])
    data = build_federated_run(cfg, 0)

    client = 2  # Parihaka site
    ds = data.client_tests[client].dataset
    k = len(ds) // 2  # representative crossline
    img, tgt, idx = ds[k]
    sec = img.squeeze(0).numpy()
    truth = tgt.numpy()

    device = "cpu"
    glob = load_global_state(run)

    # (a) averaged BN = global as-is
    m_a = UNet(1, cfg.num_classes, bilinear=False, norm=cfg.norm, groups=cfg.norm_groups)
    m_a.load_state_dict(glob)
    pred_a = predict_masked(m_a, img, device)

    # (b) local BN = same weights, BN re-estimated on this client's train
    m_b = UNet(1, cfg.num_classes, bilinear=False, norm=cfg.norm, groups=cfg.norm_groups)
    m_b.load_state_dict(glob)
    m_b = reestimate_bn(m_b, data.loaders[client], device)
    pred_b = predict_masked(m_b, img, device)

    # (d) FedBN deployed
    m_d = UNet(1, cfg.num_classes, bilinear=False, norm=cfg.norm, groups=cfg.norm_groups)
    try:
        fedbn_local = load_local_states(fedbn_run)[client]
        m_d.load_state_dict(fedbn_local)
        pred_d = predict_masked(m_d, img, device)
    except Exception:
        pred_d = None

    panels = [("(a)  averaged BN", pred_a), ("(b)  local BN", pred_b),
              ("(c)  interpretation", truth)]
    if pred_d is not None:
        panels.append(("(d)  FedBN", pred_d))

    n = len(panels)
    fig = plt.figure(figsize=(3.6 * n, 3.0))
    gs = GridSpec(1, n, figure=fig, wspace=0.05)
    for j, (title, pred) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        S.show_seismic(ax, sec)                    # grayscale seismic behind
        S.overlay_facies(ax, pred, n=cfg.num_classes, alpha=0.32, contour=True)
        ax.set_title(title, fontsize=15, color=S.NAVY, pad=8)
        if j == 0:
            ax.set_ylabel("depth", fontsize=14)
    handles = [mpatches.Patch(color=S.FACIES_COLORS[k], label=NAMES[k]) for k in range(6)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    S.save(fig, "f3_blackout_strip", (3.6 * n, 3.0))
    print("f3 seismic strip written")


if __name__ == "__main__":
    main()
