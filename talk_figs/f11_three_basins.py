"""F11 - "One shared body, three basins" (corrected).

Three crossline sections (F3, Parihaka, Penobscot) with the deployed per-survey
model's prediction over the grayscale seismic. Per-survey facies palette +
legend (each survey has its own facies). Sections cropped to a uniform window
(no squashing). Prediction vs ground truth.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import torch
from fedseismic.config import RunConfig
from fedseismic.experiment import build_federated_run
from fedseismic.models import UNet
from fedseismic.privacy.log import load_meta, load_personal_states
from talk_figs import style as S

BLOCKS = {"A": (0, 6), "B": (6, 12), "C": (12, 19)}
NCLASS = {"A": 6, "B": 6, "C": 7}
NAMES = {"A": S.F3_NAMES, "B": S.PARIHAKA_NAMES, "C": S.PENOBSCOT_NAMES}
TITLE = {"A": "F3", "B": "Parihaka", "C": "Penobscot"}


def crop_to_aspect(arr, target=1.5):
    """Centre-crop a (depth, inline) array to width:height = target (no squash)."""
    h, w = arr.shape
    if w / h > target:            # too wide -> trim width
        new_w = int(h * target)
        x0 = (w - new_w) // 2
        return arr[:, x0:x0 + new_w]
    new_h = int(w / target)       # too tall -> trim height
    y0 = (h - new_h) // 2
    return arr[y0:y0 + new_h, :]


def predict_local(model, img, block):
    model.eval()
    with torch.no_grad():
        lg = model(img.unsqueeze(0).float())
        lg = lg[0] if isinstance(lg, (tuple, list)) else lg
        lg = lg[:, block[0]:block[1]].clone()
        return lg.argmax(1).squeeze(0).numpy()   # LOCAL class index 0..K-1


def main():
    run = "results/runs/multisurvey3_fedpersurvey/fedpersurvey_noniid_a0.1_lNA_dna_n6/seed_0"
    meta = load_meta(run)
    cfg = RunConfig.from_mapping(meta["config"])
    data = build_federated_run(cfg, 0)
    pers = load_personal_states(run)
    per_survey = cfg.num_clients // 3
    picks = {"A": 0, "B": per_survey, "C": 2 * per_survey}

    fig = plt.figure(figsize=(15, 6.4))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.12, wspace=0.10)

    for j, (survey, client) in enumerate(picks.items()):
        block = BLOCKS[survey]
        nc = NCLASS[survey]
        loader = data.client_tests[client]
        ds = loader.dataset
        k = len(ds) // 2
        img, tgt, idx = ds[k]
        model = UNet(1, cfg.num_classes, bilinear=False, norm=cfg.norm, groups=cfg.norm_groups)
        model.load_state_dict(pers[client])
        pred = predict_local(model, img, block)          # local 0..nc-1
        truth = tgt.numpy() - block[0]                    # local class index
        sec = img.squeeze(0).numpy()
        sec_c = crop_to_aspect(sec, 1.5)
        pred_c = crop_to_aspect(pred, 1.5)
        truth_c = crop_to_aspect(truth, 1.5)
        cmap, _ = S.facies_cmap(nc)

        axp = fig.add_subplot(gs[0, j])
        S.show_seismic(axp, sec_c)
        S.overlay_facies(axp, pred_c, n=nc, alpha=0.5, contour=True)
        axp.set_title(f"{TITLE[survey]} — prediction", fontsize=15, color=S.NAVY, pad=6)
        if j == 0:
            axp.set_ylabel("depth", fontsize=13)

        axt = fig.add_subplot(gs[1, j])
        S.show_seismic(axt, sec_c)
        S.overlay_facies(axt, truth_c, n=nc, alpha=0.5, contour=True)
        axt.set_title(f"{TITLE[survey]} — interpretation", fontsize=13, color="#666666", pad=6)

        handles = [mpatches.Patch(color=S.FACIES_COLORS[k2], label=NAMES[survey][k2]) for k2 in range(nc)]
        axt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04),
                   ncol=2, fontsize=7.5, frameon=False, handlelength=1.2,
                   columnspacing=1.0, labelspacing=0.3)

    S.save(fig, "f11_three_basins", (15, 7.2))
    print("f11 corrected")


if __name__ == "__main__":
    main()
