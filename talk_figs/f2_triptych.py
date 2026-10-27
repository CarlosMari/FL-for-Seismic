"""F2 - section triptych per operator + prediction-over-seismic overlays.

Rows: one crossline from each operator's own holdout (the REPRESENTATIVE one -
median per-section error of that site's deployed model, not the best) and one
row from the official held-out test volume scored with the global model.
Columns: seismic | interpretation (ground truth) | FL prediction | errors
(white = correct, true-facies color = wrong, style.error_cmap).

Reuses repo code: build_federated_run (loaders/splits), models.UNet,
privacy.log.load_local_states / load_global_state (deployed checkpoints),
eval.metrics.evaluate_loader (present-class mIoU - every quoted number), and
style.* for display. Nothing is reimplemented.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
os.chdir(ROOT)

import torch
from torch.utils.data import DataLoader, Subset

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import style
from fedseismic.config import RunConfig
from fedseismic.eval import metrics
from fedseismic.experiment import build_federated_run
from fedseismic.models import UNet
from fedseismic.privacy.log import load_global_state, load_local_states, load_meta

RUN = Path("results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0")
SEED = 0
NUM_CLASSES = 6
LETTERS = "ABC"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_model(cfg):
    return UNet(1, cfg.num_classes, bilinear=False,
                norm=cfg.norm, groups=cfg.norm_groups)


def predict(model, image, device=DEVICE):
    model.eval()
    with torch.no_grad():
        logits, _ = model(image[None].to(device, dtype=torch.float))
    return logits.argmax(dim=1)[0].cpu().numpy()


def per_section_error(model, dataset, device=DEVICE):
    """1 - present-class mIoU for every crossline section of a loader dataset."""
    errors = []
    for j in range(len(dataset)):
        loader = DataLoader(Subset(dataset, [j]), batch_size=1)
        miou, _, _ = metrics.evaluate_loader(model, loader, NUM_CLASSES, device=device)
        errors.append(1.0 - miou)
    return np.asarray(errors, dtype=np.float64)


def median_index(errors):
    """The section at the median error rank (not the best one)."""
    order = np.argsort(errors)
    return int(order[len(order) // 2])


# ---------------------------------------------------------------- drawing
PW, PH = 401, 255          # train section shape (depth, inline)
PWT = 200                  # test1 section width (inline)
S = 0.0055                 # inches per pixel -> true seismic aspect everywhere
PW_IN, PH_IN = PW * S, PH * S
PWT_IN = PWT * S
GAP, GUTTER = 0.10, 1.65
VGAP = 0.12
HEADER = 0.45
LEGEND_H = 0.95


def _panel(fig, x, y, w, h):
    """Add an axes from an inch-rectangle (x, y from top-left)."""
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, (H - y - h) / H, w / W, h / H])


def draw_triptych(rows, stem):
    """rows: list of dicts with name, letter, section, target, pred."""
    ncols = 4
    W = GUTTER + ncols * PW_IN + (ncols - 1) * GAP + 0.15
    H = HEADER + len(rows) * PH_IN + (len(rows) - 1) * VGAP + LEGEND_H + 0.12
    fig = plt.figure(figsize=(W, H))
    fig.add_artist(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                 fc="white", ec="none", zorder=-10))
    hf = H  # figure height for y-flips

    headers = ["Seismic", "Interpretation", "FL prediction", "Errors"]
    for k, text in enumerate(headers):
        x = GUTTER + k * (PW_IN + GAP)
        fig.text((x + PW_IN / 2) / W, (hf - 0.22) / hf, text,
                 ha="center", va="center", fontsize=15, weight="bold",
                 color=style.NAVY)

    for i, row in enumerate(rows):
        y = HEADER + i * (PH_IN + VGAP)
        is_test = row.get("is_test", False)
        w = PWT_IN if is_test else PW_IN
        x0 = GUTTER + 0  # slot origin
        fig.text(0.12 / W, (hf - y - PH_IN / 2) / hf, row["name"], ha="left",
                 va="center", fontsize=16, weight="bold", color=style.NAVY)

        for k in range(4):
            x = GUTTER + k * (PW_IN + GAP) + (PW_IN - w) / 2
            ax = _panel(fig, x, y, w, PH_IN)
            if k == 0:
                style.show_seismic(ax, row["section"])
            elif k == 1:
                cmap, norm = style.facies_cmap(NUM_CLASSES)
                ax.imshow(row["target"], cmap=cmap, norm=norm, aspect="equal",
                          interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
            elif k == 2:
                cmap, norm = style.facies_cmap(NUM_CLASSES)
                ax.imshow(row["pred"], cmap=cmap, norm=norm, aspect="equal",
                          interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                wrong = np.where(row["pred"] == row["target"], -1, row["target"])
                cmap, norm = style.error_cmap(NUM_CLASSES)
                ax.imshow(wrong, cmap=cmap, norm=norm, aspect="equal",
                          interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])

    handles = [mpatches.Patch(color=style.FACIES_COLORS[c], label=style.F3_NAMES[c])
               for c in range(NUM_CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=14, handlelength=1.4, columnspacing=2.2, labelspacing=0.6,
               bbox_to_anchor=(0.5, 0.0))
    style.save(fig, stem)
    return {"w_in": W, "h_in": H, "aspect": W / H}


def draw_overlay(section, pred, letter, stem):
    W = PW * 0.0125
    H = PH * 0.0125 + 0.95 + 0.15
    fig = plt.figure(figsize=(W, H))
    fig.add_artist(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                 fc="white", ec="none", zorder=-10))
    ax = _panel(fig, 0.02, 0.06, PW * 0.0125, PH * 0.0125)
    style.show_seismic(ax, section)
    style.overlay_facies(ax, pred, n=NUM_CLASSES)
    style.label_panel(ax, f"Operator {letter}", size=16)
    handles = [mpatches.Patch(color=style.FACIES_COLORS[c], label=style.F3_NAMES[c])
               for c in range(NUM_CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=14, handlelength=1.4, columnspacing=1.6, labelspacing=0.5,
               bbox_to_anchor=(0.5, 0.0))
    style.save(fig, stem)


# ---------------------------------------------------------------- main
def main():
    meta = load_meta(RUN)
    cfg = RunConfig.from_mapping(meta["config"])
    data = build_federated_run(cfg, SEED)
    local_states = load_local_states(RUN)
    global_state = load_global_state(RUN)
    facts = {"run": str(RUN), "seed": SEED, "device": DEVICE, "rows": [],
             "operators": []}

    rows = []
    for i in range(3):
        model = make_model(cfg)
        model.load_state_dict(local_states[i])
        model.to(DEVICE).eval()
        ds = data.client_tests[i].dataset          # InlineLoader over holdout tail
        errors = per_section_error(model, ds)
        j = median_index(errors)
        image, target, _ = ds[j]
        pred = predict(model, image)
        section = image[0].numpy()
        target = target.numpy()
        miou_local = metrics.evaluate_loader(
            model, data.client_tests[i], NUM_CLASSES, device=DEVICE)[0]
        rows.append({"name": f"Operator {LETTERS[i]}", "letter": LETTERS[i],
                     "section": section, "target": target, "pred": pred})
        draw_overlay(section, pred, LETTERS[i], f"f2_overlay_{LETTERS[i]}")
        facts["operators"].append({
            "operator": f"Operator {LETTERS[i]}", "client": i,
            "n_holdout": len(ds),
            "chosen_xline": int(ds.indices[j]),
            "chosen_rank": "median of %d holdout sections by per-section "
                           "error (1 - present-class mIoU), of the site's own "
                           "deployed model local/client_%03d.pt" % (len(ds), i),
            "chosen_section_miou": round(1.0 - float(errors[j]), 4),
            "error_median": round(float(np.median(errors)), 4),
            "error_min": round(float(errors.min()), 4),
            "error_max": round(float(errors.max()), 4),
            "local_test_miou_deployed": round(float(miou_local), 4),
        })
        print(f"  Operator {LETTERS[i]}: holdout {len(ds)} sections, "
              f"median-error crossline {ds.indices[j]} "
              f"(err {np.median(errors):.3f}, range {errors.min():.3f}-{errors.max():.3f}), "
              f"local mIoU {miou_local:.3f}")

    # official held-out test volume with the global model
    if global_state is not None and data.tests.get("test1") is not None:
        model = make_model(cfg)
        model.load_state_dict(global_state)
        model.to(DEVICE).eval()
        test_loader, test_labels = data.tests["test1"]
        ds = test_loader.dataset
        errors = per_section_error(model, ds)
        j = median_index(errors)
        image, target, _ = ds[j]
        pred = predict(model, image)
        miou_test, per_class_test, _ = metrics.evaluate_loader(
            model, test_loader, NUM_CLASSES, device=DEVICE)
        test2 = data.tests.get("test2")
        miou_test2 = (metrics.evaluate_loader(
            model, test2[0], NUM_CLASSES, device=DEVICE)[0]
            if test2 is not None else None)
        rows.append({"name": "Official test", "letter": "T", "is_test": True,
                     "section": image[0].numpy(), "target": target.numpy(),
                     "pred": pred})
        facts["test_row"] = {
            "volume": "datasets/seismic/data/test_once/test1_seismic.npy",
            "model": "global_final.pt",
            "n_sections": len(ds),
            "chosen_xline": int(ds.indices[j]),
            "chosen_rank": "median of %d test sections by per-section error" % len(ds),
            "chosen_section_miou": round(1.0 - float(errors[j]), 4),
            "error_median": round(float(np.median(errors)), 4),
            "test_miou_global": round(float(miou_test), 4),
            "test_per_class_iou": [round(float(v), 4) for v in per_class_test],
            "test2_miou_global": None if miou_test2 is None else round(float(miou_test2), 4),
        }
        print(f"  Official test: {len(ds)} sections, median-error crossline "
              f"{ds.indices[j]}, test mIoU {miou_test:.3f}")

    facts["triptych"] = draw_triptych(rows, "f2_triptych_n3")
    (HERE / "_facts_f2.json").write_text(json.dumps(facts, indent=1),
                                         encoding="utf-8")
    write_facts_md(facts)
    print("  wrote talk_figs/_facts_f2.json and talk_figs/facts_F1F2.md")


def write_facts_md(f2):
    f1_path = HERE / "_facts_f1.json"
    f1 = json.loads(f1_path.read_text(encoding="utf-8")) if f1_path.exists() else {}

    def pct(vec):
        return " ".join(f"{100*v:.1f}%" for v in vec)

    lines = []
    lines.append("# F1/F2 facts (sponsor talk)")
    lines.append("")
    lines.append(f"- Run: `{f2['run']}` (fedavg, noniid contiguous blocks, "
                 f"a=0.1, n=3, batchnorm), seed {f2['seed']}.")
    lines.append(f"- Seed: {f2['seed']} (all splits and crossline choices are "
                 f"deterministic in this seed).")
    lines.append(f"- Deployed models: `local/client_000.pt` - `local/client_002.pt` "
                 f"(each site's own BN); official test row uses `global_final.pt`.")
    lines.append(f"- Metrics quoted are present-class mIoU "
                 f"(`fedseismic.eval.metrics.evaluate_loader`).")
    lines.append("")
    lines.append("## F1 - survey split into operator blocks")
    for key, label in (("n3", "N=3"), ("n5", "N=5")):
        run = f1.get("runs", {}).get(key)
        if not run:
            continue
        lines.append(f"### {label} split ({run['run']})")
        lines.append("")
        lines.append("| operator | crosslines | block | train | holdout tail "
                     "| facies mix (train pixels, 0..5) | lacks Zechstein |")
        lines.append("|---|---|---|---|---|---|---|")
        for s in run["stats"]:
            lacks = "yes (0 class-5 px)" if s["lacks_c5_full"] else \
                ("in train split" if s["lacks_c5_train"] else "no")
            lines.append(
                f"| Operator {s['letter']} | {s['n_xlines']} | "
                f"{s['block'][0]}-{s['block'][1]} | {s['train'][0]}-{s['train'][1]} | "
                f"{s['holdout_tail'][0]}-{s['holdout_tail'][1]} | "
                f"{pct(s['pi_train'])} | {lacks} |")
        lines.append("")
        zero_full = "".join(s["letter"] for s in run["stats"] if s["lacks_c5_full"])
        zero_train = "".join(s["letter"] for s in run["stats"] if s["lacks_c5_train"])
        n = len(run["stats"])
        lines.append(f"- Operators with **zero** Zechstein (class 5) pixels "
                     f"anywhere in their block: **{run['n_lacking_c5_full']} of "
                     f"{n}** (operator {', '.join(zero_full)}).")
        lines.append(f"- Operators with zero Zechstein in their *training* split "
                     f"(the pi histograms): **{run['n_lacking_c5_train']} of "
                     f"{n}** (operator {', '.join(zero_train)}).")
        for s in run["stats"]:
            if s["lacks_c5_train"] and not s["lacks_c5_full"]:
                lines.append(
                    f"- Note: operator {s['letter']}'s only Zechstein pixels "
                    f"({100 * s['c5_pixels_full']:.2f}% of its block) lie in its "
                    f"holdout tail (xline {s['holdout_tail'][0]}-"
                    f"{s['holdout_tail'][1]}); its training split has none.")
    lines.append("")
    lines.append("## F2 - section triptych (chosen crosslines)")
    lines.append("")
    lines.append("Representative crossline = median per-section error "
                 "(1 - present-class mIoU) of the row's own model over its "
                 "whole holdout - not the best section. Global row: median over "
                 "all 701 crosslines of test1.")
    lines.append("")
    lines.append("| row | crossline | why | section mIoU | holdout mIoU |")
    lines.append("|---|---|---|---|---|")
    for op in f2["operators"]:
        lines.append(f"| {op['operator']} (client {op['client']}, "
                     f"local/client_{op['client']:03d}.pt) | {op['chosen_xline']} | "
                     f"median error of {op['n_holdout']} holdout sections "
                     f"(err median {op['error_median']:.3f}, "
                     f"range {op['error_min']:.3f}-{op['error_max']:.3f}) | "
                     f"{op['chosen_section_miou']:.3f} | "
                     f"{op['local_test_miou_deployed']:.3f} |")
    tr = f2.get("test_row")
    if tr:
        lines.append(f"| Official test (global_final.pt) | {tr['chosen_xline']} | "
                     f"median error of {tr['n_sections']} test1 sections | "
                     f"{tr['chosen_section_miou']:.3f} | "
                     f"{tr['test_miou_global']:.3f} (all test1) |")
    lines.append("")
    if tr:
        pc = tr.get("test_per_class_iou")
        lines.append(f"- Global model on the official test volume: "
                     f"present-class mIoU = {tr['test_miou_global']:.3f}"
                     + (f"; per-class IoU {' '.join('%.3f' % v for v in pc)}."
                        if pc else ".")
                     + " Zechstein (class 5) is present in the official test "
                       "but never recovered (IoU 0.000).")
        if tr.get("test2_miou_global") is not None:
            lines.append(f"- The run's logged score (results.json `final` = "
                         f"0.568) is the mean of the test1 and test2 "
                         f"evaluations; with the saved `global_final.pt`, "
                         f"test1 = {tr['test_miou_global']:.3f} and test2 = "
                         f"{tr['test2_miou_global']:.3f} "
                         f"(mean {(tr['test_miou_global'] + tr['test2_miou_global']) / 2:.3f}).")
        lines.append("- Talk hook: operator B's chosen section (xline 464) is a "
                     "salt diapir with a Zechstein core - B's only Zechstein "
                     "lives in its holdout tail - and the deployed model calls "
                     "it Scruff/Rijnland (see the vermillion error blob).")
    lines.append(f"- Figure aspect: f2_triptych_n3 = "
                 f"{f2['triptych']['aspect']:.2f}:1 (4 rows x 4 columns of "
                 f"true-aspect 401x255 panels; a 3:1 overall figure is not "
                 f"geometrically possible without squashing the seismic).")
    lines.append("")
    (HERE / "facts_F1F2.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
