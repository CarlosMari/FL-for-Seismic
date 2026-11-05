"""Fig 6.5: Parihaka and the F3 control under averaged, re-estimated and FedBN BN."""
from thesis_ch6_common import *

import torch

from f3_blackout import (BLOCK_F3, BLOCK_PARIHAKA, build_model, masked_argmax, per_sample_miou, refresh_bn,
                         score_masked, score_unmasked)
from fedseismic.config import RunConfig
from fedseismic.eval.metrics import _logits
from fedseismic.experiment import build_federated_run
from fedseismic.privacy.log import load_local_states, load_meta

FEDAVG = ROOT / "results/runs/multisurvey/fedavg_noniid_a0.1_lNA_dna_n4/seed_0"
FEDBN = ROOT / "results/runs/multisurvey_fedbn/fedbn_noniid_a0.1_lNA_dna_n4/seed_0"
CACHE = DATA / "fig6_5_sections.npz"
GROUPS = {0: BLOCK_F3, 1: BLOCK_F3, 2: BLOCK_PARIHAKA, 3: BLOCK_PARIHAKA}


def local_pred(model, image, block):
    with torch.no_grad():
        logits = _logits(model(image[None].to("cpu", dtype=torch.float)))
    return masked_argmax(logits, block)[0].numpy() - block[0]


def compute():
    torch.set_num_threads(EVAL_THREADS)
    meta = load_meta(FEDAVG)
    cfg = RunConfig.from_mapping(meta["config"])
    data = build_federated_run(cfg, int(meta.get("seed", cfg.seed)))
    state = torch.load(FEDAVG / "global_final.pt", map_location="cpu", weights_only=True)
    fedbn_states = load_local_states(FEDBN)
    model_a = build_model(); model_a.load_state_dict(state); model_a.eval()
    scores, arrays, picks, models_b, models_c = {}, {}, {}, {}, {}
    for c in range(4):
        models_b[c] = build_model(); models_b[c].load_state_dict(state)
        scores[f"bn_batches_c{c}"] = refresh_bn(models_b[c], data.loaders[c])
        models_c[c] = build_model(); models_c[c].load_state_dict(fedbn_states[c]); models_c[c].eval()
    for c in range(4):
        loader, block = data.client_tests[c], GROUPS[c]
        for name, model in (("a", model_a), ("b", models_b[c]), ("c", models_c[c])):
            scores[f"{name}_c{c}_masked"] = score_masked(model, loader, block)[0]
            scores[f"{name}_c{c}_unmasked"] = score_unmasked(model, loader)[0]
            print(name, c, scores[f"{name}_c{c}_masked"], scores[f"{name}_c{c}_unmasked"], flush=True)
    for tag, c in (("pari", 2), ("f3", 0)):
        loader, block = data.client_tests[c], GROUPS[c]
        rows = sorted(per_sample_miou(models_b[c], loader, block), key=lambda r: r[1])
        xl, section_b = rows[len(rows) // 2]
        ds = loader.dataset
        j = list(ds.indices).index(xl)
        image, target, _ = ds[j]
        target = target.numpy()
        arrays[f"{tag}_section"] = image[0].numpy()
        arrays[f"{tag}_truth"] = np.where(target == 255, 255, target - block[0])
        arrays[f"{tag}_a"] = local_pred(model_a, image, block)
        arrays[f"{tag}_b"] = local_pred(models_b[c], image, block)
        arrays[f"{tag}_c"] = local_pred(models_c[c], image, block)
        picks[tag] = {"client": c, "crossline": int(xl), "n_holdout": len(rows), "section_miou_b": float(section_b)}
    write_facts("fig6_5", {"scores": scores, "picks": picks,
                           "survey_b_seismic": str(cfg.survey_b_seismic)})
    np.savez_compressed(CACHE, **arrays)


def draw():
    a = np.load(CACHE)
    facts = json.loads((DATA / "fig6_5.json").read_text())
    scores, picks = facts["scores"], facts["picks"]
    # Two-survey Parihaka was trained with reflectors vertical; display only is rotated.
    rot = {"pari": np.rot90, "f3": lambda x: x}
    gutter, gap = 0.2, 0.04
    pw = (W - gutter - 4 * gap - 0.02) / 5
    y = 0.2
    fig_h = 0.2
    heights = {}
    for tag in ("pari", "f3"):
        shape = rot[tag](a[f"{tag}_section"]).shape
        heights[tag] = pw * shape[0] / shape[1]
    H = 0.2 + heights["pari"] + 0.07 + heights["f3"] + 0.5
    fig = plt.figure(figsize=(W, H))
    heads = ("Seismic", "Ground truth", "(a) FedAvg, averaged BN", "(b) re-estimated BN", "(c) FedBN deployed")
    for k, text in enumerate(heads):
        fig.text((gutter + k * (pw + gap) + pw / 2) / W, 1 - 0.1 / H, text, ha="center", va="center", fontsize=8)
    for tag, name, c in (("pari", "Parihaka", 2), ("f3", "F3 control", 0)):
        h = heights[tag]; f = rot[tag]
        sec = f(a[f"{tag}_section"])
        fig.text(0.1 / W, 1 - (y + h / 2) / H, f"{name}\nxl {picks[tag]['crossline']}", rotation=90,
                 ha="center", va="center", fontsize=8, linespacing=0.95)
        for k, key in enumerate(("section", "truth", "a", "b", "c")):
            ax = inch_axes(fig, gutter + k * (pw + gap), y, pw, h)
            if key == "section":
                seismic(ax, sec)
            else:
                facies_over(ax, sec, f(a[f"{tag}_{key}"]))
            if key in "abc":
                panel_label(ax, f"({key})")
                corner_note(ax, f"{scores[f'{key}_c{c}_masked']:.2f}")
        y += h + 0.07
    legends = (("F3", style.F3_NAMES, 6, 0.33), ("Parihaka", style.PARIHAKA_NAMES, 3, 0.17))
    for label, names, ncol, y_in in legends:
        fig.legend(handles=facies_handles(names), loc="center left", ncol=ncol, frameon=False,
                   bbox_to_anchor=(0.62 / W, (y_in + (0.0 if ncol == 6 else -0.02)) / H), handlelength=1.0,
                   handletextpad=0.25, columnspacing=0.7, borderpad=0, labelspacing=0.15)
        fig.text(0.05 / W, (y_in - (0.0 if ncol == 6 else 0.02)) / H, f"{label}:", ha="left", va="center", fontsize=8)
    print("fig6_5 height", H, heights)
    save(fig, "fig6_5_bn_blackout")


if __name__ == "__main__":
    if not CACHE.is_file():
        compute()
    draw()
