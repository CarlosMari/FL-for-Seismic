"""Fig 6.2: section triptych for three F3 sites plus the official test volume."""
from thesis_ch6_common import *

import torch

from f2_triptych import median_index, per_section_error, predict
from fedseismic.config import RunConfig
from fedseismic.eval import metrics
from fedseismic.experiment import build_federated_run
from fedseismic.models import UNet
from fedseismic.privacy.log import load_global_state, load_local_states, load_meta

RUN = ROOT / "results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0"
CACHE = DATA / "fig6_2_sections.npz"


def compute():
    torch.set_num_threads(EVAL_THREADS)
    cfg = RunConfig.from_mapping(load_meta(RUN)["config"])
    data = build_federated_run(cfg, 0)
    states = load_local_states(RUN)

    def model_of(state):
        model = UNet(1, cfg.num_classes, bilinear=False, norm=cfg.norm, groups=cfg.norm_groups)
        model.load_state_dict(state)
        return model.to("cpu").eval()

    rows, arrays = [], {}

    def add(tag, name, model, loader):
        ds = loader.dataset
        errors = per_section_error(model, ds, device="cpu")
        j = median_index(errors)
        image, target, _ = ds[j]
        miou, per_class, _ = metrics.evaluate_loader(model, loader, 6, device="cpu")
        rows.append({"row": name, "crossline": int(ds.indices[j]), "n_sections": len(ds),
                     "section_miou": 1 - float(errors[j]), "miou": float(miou),
                     "per_class_iou": [float(v) for v in per_class],
                     "error_min": float(errors.min()), "error_max": float(errors.max())})
        arrays[f"{tag}_section"] = image[0].numpy()
        arrays[f"{tag}_truth"] = target.numpy()
        arrays[f"{tag}_pred"] = predict(model, image, device="cpu")
        print(rows[-1], flush=True)

    for i in range(3):
        add(f"s{i}", f"Site {LETTERS[i]}", model_of(states[i]), data.client_tests[i])
    add("test", "Official test1", model_of(load_global_state(RUN)), data.tests["test1"][0])
    write_facts("fig6_2", {"rows": rows})
    np.savez_compressed(CACHE, **arrays)


def draw():
    a = np.load(CACHE)
    rows = json.loads((DATA / "fig6_2.json").read_text())["rows"]
    gutter = 0.42; gap = 0.05
    pw = (W - gutter - 3 * gap - 0.03) / 4
    scale = pw / a["s0_section"].shape[1]
    ph = a["s0_section"].shape[0] * scale
    head, vgap = 0.2, 0.2
    H = head + 4 * ph + 3 * vgap + 0.42
    fig = plt.figure(figsize=(W, H))
    for k, text in enumerate(("Seismic", "Ground truth", "Deployed prediction", "Error (true facies)")):
        fig.text((gutter + k * (pw + gap) + pw / 2) / W, 1 - 0.1 / H, text, ha="center", va="center")
    for i, tag in enumerate(("s0", "s1", "s2", "test")):
        sec, truth, pred = a[f"{tag}_section"], a[f"{tag}_truth"], a[f"{tag}_pred"]
        w = sec.shape[1] * scale; h = sec.shape[0] * scale
        y = head + i * (ph + vgap)
        row = rows[i]
        fig.text(0.16 / W, 1 - (y + h / 2) / H,
                 "Test1" if tag == "test" else row["row"], rotation=90, ha="center", va="center")
        fig.text(0.31 / W, 1 - (y + h / 2) / H, f"xl {row['crossline']}", rotation=90, ha="center", va="center", fontsize=8)
        axes = []
        for k in range(4):
            x = gutter + k * (pw + gap) + (pw - w) / 2
            ax = inch_axes(fig, x, y, w, h)
            axes.append(ax)
            if k == 0:
                seismic(ax, sec)
                panel_label(ax, f"({'abcd'[i]})")
            elif k == 1:
                facies_over(ax, sec, truth)
            elif k == 2:
                facies_over(ax, sec, pred)
            else:
                error_map(ax, truth, pred)
        fig.text((gutter + pw * 2 + gap * 2) / W, 1 - (y + h + 0.015) / H,
                 f"{'official test1' if tag == 'test' else 'holdout'} mIoU {row['miou']:.3f}; this section {row['section_miou']:.3f}",
                 ha="center", va="top", fontsize=8)
        if tag == "s1":
            zr, zc = np.where(truth == 5)
            xy = (float(zc.mean()), float(zr.mean()))
            rows[i]["zechstein_section_px"] = int(zr.size)
            for ax in (axes[1], axes[3]):
                ax.annotate("Zechstein core", xy=xy, xytext=(0.04, 0.42), textcoords="axes fraction", fontsize=8,
                            arrowprops=dict(arrowstyle="-|>", lw=0.7, color="k", shrinkB=1, mutation_scale=6),
                            bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))
    fig.legend(handles=facies_handles(style.F3_NAMES), loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.1, handletextpad=0.3, columnspacing=0.9)
    print("fig6_2 height", H)
    save(fig, "fig6_2_triptych")
    write_facts("fig6_2", {"rows": rows})


if __name__ == "__main__":
    if not CACHE.is_file():
        compute()
    draw()
