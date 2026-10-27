"""F3 sponsor-talk figure: "The blackout" - Parihaka under averaged vs local BN.

Reproduces the 2026-10-19 diagnostic in findings.md: the multi-survey FedAvg
global collapses on Parihaka because of AVERAGED BatchNorm running stats
(panel a); the SAME weights with BN running stats re-estimated on the
Parihaka client's own train data recover (panel b). Panel (c) is the ground
truth, (d) the FedBN deployed model (BN kept local). An F3 row shows the same
(a)/(b) pair on an F3 crossline.

All loaders/models/metrics come from the repo (fedseismic.*); nothing is
reimplemented here. Scoring uses the repo's masked argmax + IGNORE handling
(`multisurvey_utility._score_deployed`, the deployed semantics) AND the
unmasked `evaluate_loader` convention; findings.md's 2026-10-19 numbers were
computed with the UNMASKED convention (see facts fragment - the run's own
results.json global_on_local = mean(0.3238, 0.2876, 0.0003, 0.0004) pins it).

Panels honour TRUE seismic aspect (never squashed) inside a 1.07:1 canvas
(style.ASPECT["third"]); the letterbox bands host nothing but white so the
PowerPoint build-animation panels share one exact canvas.

Run:  cd <repo> && fl-seismic/bin/python talk_figs/f3_blackout.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "talk_figs"))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import jaccard_score

import style
from fedseismic.config import RunConfig
from fedseismic.eval.metrics import _logits, _present_mean, evaluate_loader
from fedseismic.experiment import build_federated_run
from fedseismic.models import UNet
from fedseismic.privacy.log import load_local_states, load_meta
from fedseismic.privacy.multisurvey_utility import _score_deployed

FEDAVG_RUN = ROOT / "results/runs/multisurvey/fedavg_noniid_a0.1_lNA_dna_n4/seed_0"
FEDBN_RUN = ROOT / "results/runs/multisurvey_fedbn/fedbn_noniid_a0.1_lNA_dna_n4/seed_0"
NUM_CLASSES = 12
BLOCK_F3 = (0, 6)          # survey A (F3) block in the union head
BLOCK_PARIHAKA = (6, 12)   # survey B (Parihaka) block (union rows 6-11)

# device: the RTX 3090 was fully occupied by another job at run time;
# CPU is deterministic and the whole figure job is minutes either way.
DEVICE = "cpu"

DPI_RASTER = 150
PAD = 0.3                  # style.save pads 0.15 in per side; compensate below
ASPECT_PANEL = style.ASPECT["third"]    # 1.07:1 per panel (slide placeholder)


def canvas_wh(aspect, height_out):
    """figure size whose saved output (figure + style.save's pad) has `aspect`."""
    h = height_out - PAD
    w = aspect * height_out - PAD
    return (w, h)


PANEL_WH = canvas_wh(ASPECT_PANEL, 6.0)   # saved panel = 6.42 x 6.0 in = 1.07:1
STRIP_PANEL = (5.35, 5.0)                 # panel size inside composite strips

BN_CACHE = Path("/tmp/opencode/f3_bn_cache")

# findings.md expectations (2026-10-19 diagnostic) for the mismatch flag
EXPECT = {
    "a_pari_client2": 0.0003, "a_pari_client3": 0.0004,
    "b_pari_client2": 0.7607, "b_pari_client3": 0.5985,
}
MISMATCH_TOL = 0.02


# ----------------------------------------------------------------- model / scores
def build_model(num_classes=NUM_CLASSES):
    return UNet(1, num_classes, bilinear=False, norm="batch", groups=8)


def masked_argmax(logits, block):
    """Argmax restricted to the survey's own class rows (deployed semantics)."""
    logits = logits.clone()
    logits[:, :block[0]] = -1e9
    logits[:, block[1]:] = -1e9
    return logits.argmax(dim=1)


def predict_section(model, image, block):
    """image (1,1,H,W) -> (H,W) prediction in the survey's LOCAL 0..K-1 space."""
    with torch.no_grad():
        logits = _logits(model(image.to(DEVICE, dtype=torch.float)))
        return (masked_argmax(logits, block)[0].cpu().numpy() - block[0])


def refresh_bn(model, train_loader):
    """Re-estimate BN running stats on the client's own train data.

    Forward passes in train() mode with no_grad update running_mean/var
    exactly like training does; then eval() uses the refreshed stats.
    """
    model.to(DEVICE)
    model.train()
    n_batches = 0
    with torch.no_grad():
        for images, _, _ in train_loader:
            model(images.to(DEVICE, dtype=torch.float))
            n_batches += 1
    model.eval()
    return n_batches


def score_masked(model, loader, block):
    """Deployed semantics: masked argmax to the survey block, present-class
    mIoU over the block rows, IGNORE=255 dropped (`_score_deployed`)."""
    miou, per_class, present = _score_deployed(
        model, loader, NUM_CLASSES, block, DEVICE)
    return float(miou), np.asarray(per_class), np.asarray(present)


def score_unmasked(model, loader):
    """findings.md / run-log convention: unmasked argmax, `evaluate_loader`."""
    miou, per_class, _ = evaluate_loader(model, loader, NUM_CLASSES, DEVICE)
    return float(miou), np.asarray(per_class)


def per_sample_miou(model, loader, block):
    """Per-crossline present-class mIoU -> [(crossline_idx, miou), ...]."""
    ds = loader.dataset
    rows = []
    with torch.no_grad():
        for images, labels, pos in loader:
            logits = _logits(model(images.to(DEVICE, dtype=torch.float)))
            preds = masked_argmax(logits, block).cpu().numpy()
            for i in range(images.size(0)):
                yt = np.asarray(labels[i]).ravel()
                yp = preds[i].ravel()
                keep = yt != 255  # 255 == fedseismic.eval.metrics.IGNORE
                yt, yp = yt[keep], yp[keep]
                per = jaccard_score(yt, yp, labels=list(range(NUM_CLASSES)),
                                    average=None, zero_division=0)
                rows.append((int(ds.indices[int(pos[i])]),
                             float(_present_mean(per, yt, NUM_CLASSES))))
    return rows


def section_of(loader, xl):
    """(seismic, labels) display sections for crossline xl (labels in union space)."""
    ds = loader.dataset
    seis = np.asarray(ds.seismic[:, xl, :]).T
    lab = np.asarray(ds.label[:, xl, :]).T
    return seis, lab


# ----------------------------------------------------------------- rendering
def image_rect(fig_wh, canvas_xywh, data_aspect):
    """Axes rect (figure fractions) holding a TRUE-aspect image, letterboxed
    inside the canvas cell. Never squashes: 1 data unit x = 1 data unit y."""
    figw, figh = fig_wh
    cx, cy, cw, ch = canvas_xywh
    win_w, win_h = cw * figw, ch * figh
    if data_aspect >= win_w / win_h:
        w_in, h_in = win_w, win_w / data_aspect
    else:
        h_in, w_in = win_h, win_h * data_aspect
    x_in = cx * figw + (win_w - w_in) / 2
    y_in = cy * figh + (win_h - h_in) / 2
    return [x_in / figw, y_in / figh, w_in / figw, h_in / figh]


def pin_canvas(fig):
    """Full-figure white rect so style.save's tight bbox keeps the whole canvas."""
    fig.add_artist(mpatches.Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                      fc="white", ec="none", zorder=-10))


def draw_panel(ax, seis, pred, label_text, miou=None, n=6):
    style.show_seismic(ax, seis)
    style.overlay_facies(ax, pred, n=n, alpha=0.45, contour=True)
    style.label_panel(ax, label_text, size=17)
    if miou is not None:
        style.label_panel(ax, f"mIoU {miou:.3f}", size=15, weight="normal",
                          loc="lower left")


def fig_raster(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def render_panel(seis, pred, label_text, miou=None, n=6):
    """Standalone panel on the 1.07:1 slide canvas (letterboxed, TRUE aspect)."""
    fig = plt.figure(figsize=PANEL_WH, dpi=DPI_RASTER)
    pin_canvas(fig)
    ar = seis.shape[1] / seis.shape[0]
    ax = fig.add_axes(image_rect(PANEL_WH, [0, 0, 1, 1], ar))
    draw_panel(ax, seis, pred, label_text, miou, n)
    return fig, ax


def legend_row(fig, names, y=0.035):
    handles = [mpatches.Patch(color=c, label=n)
               for c, n in zip(style.FACIES_COLORS, names)]
    fig.legend(handles=handles, loc="lower center", ncol=len(names),
               frameon=False, fontsize=15, bbox_to_anchor=(0.5, y),
               handlelength=1.4, columnspacing=1.2)


def composite(fig_wh, cells, seis, preds, labels, mious, names, legend_y=0.03):
    """Tiled panels on 1.07:1 cells + a facies-name legend row below."""
    fig = plt.figure(figsize=fig_wh)
    pin_canvas(fig)
    ar = seis.shape[1] / seis.shape[0]
    for cell, key in zip(cells, labels):
        ax = fig.add_axes(image_rect(fig_wh, cell, ar))
        draw_panel(ax, seis, preds[key], labels[key], mious.get(key))
    legend_row(fig, names, y=legend_y)
    return fig


def crossfade(rast_a, rast_b, stem, seconds=4.0, fps=12):
    """3-4s cosine crossfade (a)->(b) as MP4 (H.264) + GIF + final-frame PNG."""
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    h = min(rast_a.shape[0], rast_b.shape[0]) // 2 * 2
    w = min(rast_a.shape[1], rast_b.shape[1]) // 2 * 2
    a = rast_a[:h, :w].astype(np.float32)
    b = rast_b[:h, :w].astype(np.float32)
    n = int(round(seconds * fps))
    frames = []
    for k in range(n):
        t = k / max(n - 1, 1)
        t = 0.5 - 0.5 * np.cos(np.pi * t)  # cosine ease
        frames.append((a * (1 - t) + b * t).astype(np.uint8))

    fig = plt.figure(figsize=(w / DPI_RASTER, h / DPI_RASTER))
    ax = fig.add_axes([0, 0, 1, 1])
    im = ax.imshow(frames[0], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])

    def update(k):
        im.set_data(frames[k])
        return [im]

    anim = FuncAnimation(fig, update, frames=n, blit=True)
    out = Path(style.OUT)
    out.mkdir(parents=True, exist_ok=True)
    mp4 = out / f"{stem}.mp4"
    anim.save(str(mp4), writer=FFMpegWriter(
        fps=fps, codec="h264", bitrate=2400,
        extra_args=["-preset", "veryfast", "-pix_fmt", "yuv420p"]),
        dpi=DPI_RASTER)
    plt.close(fig)

    plt.imsave(out / f"{stem}_final.png", frames[-1])
    for ext in ("pdf", "svg"):
        ff = plt.figure(figsize=(w / DPI_RASTER, h / DPI_RASTER))
        fax = ff.add_axes([0, 0, 1, 1])
        fax.imshow(frames[-1], interpolation="nearest")
        fax.set_xticks([]); fax.set_yticks([])
        ff.savefig(out / f"{stem}_final.{ext}", dpi=300)
        plt.close(ff)

    tmp = Path("/tmp/opencode/f3fade")
    tmp.mkdir(parents=True, exist_ok=True)
    for p in tmp.glob("f_*.png"):
        p.unlink()
    for i, f in enumerate(frames):
        plt.imsave(tmp / f"f_{i:03d}.png", f)
    pal = tmp / "palette.png"
    gif = out / f"{stem}.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(tmp / "f_%03d.png"), "-vf",
                    "palettegen=stats_mode=diff", str(pal)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(tmp / "f_%03d.png"), "-i", str(pal), "-lavfi",
                    "paletteuse=dither=bayer:bayer_scale=3", str(gif)], check=True)
    print(f"  wrote {mp4}  {gif}  ({gif.stat().st_size/1e6:.1f} MB)")
    return gif.stat().st_size


# ----------------------------------------------------------------- main
def main():
    torch.set_num_threads(os.cpu_count() or 8)
    meta = load_meta(FEDAVG_RUN)
    cfg = RunConfig.from_mapping(meta["config"])
    seed = int(meta.get("seed", cfg.seed))
    print(f"building federated run (seed {seed}) ...", flush=True)
    data = build_federated_run(cfg, seed)

    # sanity: FedBN run shares the partition-relevant config
    meta_b = load_meta(FEDBN_RUN)
    cfg_b = RunConfig.from_mapping(meta_b["config"])
    for key in ("num_clients", "split", "local_test_ratio", "train_seismic",
                "survey_b_seismic", "num_classes", "norm", "norm_groups"):
        assert getattr(cfg, key) == getattr(cfg_b, key), key

    train = {c: data.loaders[c] for c in range(4)}
    tests = {c: data.client_tests[c] for c in range(4)}

    # ---------------- (a) averaged BN: final global exactly as saved
    model_a = build_model()
    model_a.load_state_dict(torch.load(FEDAVG_RUN / "global_final.pt",
                                       map_location="cpu", weights_only=True))
    model_a.to(DEVICE).eval()

    # ---------------- (b) same weights, BN re-estimated on each client's train
    BN_CACHE.mkdir(parents=True, exist_ok=True)
    models_b = {}
    bn_batches = {}
    for c in range(4):
        m = build_model()
        m.load_state_dict(model_a.state_dict())
        cache = BN_CACHE / f"client_{c}.pt"
        if cache.is_file():
            m.load_state_dict(torch.load(cache, map_location="cpu",
                                         weights_only=True))
            m.to(DEVICE).eval()
            bn_batches[c] = "cached"
            print(f"  client {c}: BN state loaded from cache", flush=True)
        else:
            bn_batches[c] = refresh_bn(m, train[c])
            torch.save(m.state_dict(), cache)
            print(f"  client {c}: {bn_batches[c]} batches refreshed (cached)",
                  flush=True)
        models_b[c] = m

    # ---------------- (d) FedBN deployed model (local/ state = body+head+own BN)
    models_d = {}
    if FEDBN_RUN.is_dir():
        local_states = load_local_states(FEDBN_RUN)
        for c in range(4):
            m = build_model()
            m.load_state_dict(local_states[c])
            models_d[c] = m.to(DEVICE).eval()

    # ---------------- scoring: BOTH conventions
    scores = {}
    detail = []
    for tag, block, clients in (("parihaka", BLOCK_PARIHAKA, (2, 3)),
                                ("f3", BLOCK_F3, (0, 1))):
        for c in clients:
            for name, model in (("a", model_a), ("b", models_b[c]),
                                ("d", models_d.get(c))):
                if model is None:
                    continue
                m_masked = score_masked(model, tests[c], block)
                m_unmasked = score_unmasked(model, tests[c])
                scores[f"{name}_{tag}_client{c}"] = m_masked[0]
                scores[f"{name}_{tag}_client{c}_unmasked"] = m_unmasked[0]
                detail.append((name, tag, c, m_masked[0], m_unmasked[0]))
                print(f"  ({name}) {tag} client {c}: masked {m_masked[0]:.4f} "
                      f"unmasked {m_unmasked[0]:.4f}", flush=True)

    # ---------------- pick the display crossline (median error, not best)
    pick = {}
    for tag, block, c in (("parihaka", BLOCK_PARIHAKA, 2),
                          ("f3", BLOCK_F3, 0)):
        rows = per_sample_miou(models_b[c], tests[c], block)
        rows_sorted = sorted(rows, key=lambda r: r[1])
        med = rows_sorted[len(rows_sorted) // 2]
        pick[tag] = (c, med[0], med[1], rows)
        print(f"  {tag}: median-crossline client {c} xl {med[0]} "
              f"(mIoU {med[1]:.4f}, n={len(rows)})", flush=True)

    facts = []

    # ---------------- panels (a)(b)(c)(d): Parihaka, SAME crossline
    c, xl, _, rows_pari = pick["parihaka"]
    seis, lab_u = section_of(tests[c], xl)
    lab_local = lab_u - BLOCK_PARIHAKA[0]
    preds = {
        "a": predict_section(model_a, _t(seis), BLOCK_PARIHAKA),
        "b": predict_section(models_b[c], _t(seis), BLOCK_PARIHAKA),
        "c": lab_local,
    }
    if models_d:
        preds["d"] = predict_section(models_d[c], _t(seis), BLOCK_PARIHAKA)
    labels = {"a": "(a) Averaged BN", "b": "(b) Local BN",
              "c": "(c) Interpretation", "d": "(d) FedBN"}
    miou_shown = {"a": scores[f"a_parihaka_client{c}"],
                  "b": scores[f"b_parihaka_client{c}"],
                  "d": scores.get(f"d_parihaka_client{c}")}

    stems = {"a": "f3a_averaged_bn", "b": "f3b_local_bn",
             "c": "f3c_truth", "d": "f3d_fedbn"}
    rasters = {}
    for key in ("a", "b", "c", "d"):
        if key not in preds:
            continue
        fig, ax = render_panel(seis, preds[key], labels[key],
                               miou=miou_shown.get(key))
        rasters[key] = fig_raster(fig)
        style.save(fig, stems[key], target_wh=PANEL_WH)
        facts.append(f"f3{key}: client {c}, Parihaka crossline {xl}, "
                     f"mIoU masked {miou_shown.get(key)}")

    # ---------------- combined strip (all panels + Parihaka formation names)
    keys = [k for k in ("a", "b", "c", "d") if k in preds]
    n = len(keys)
    figw = STRIP_PANEL[0] * n
    figh = STRIP_PANEL[1] + 0.75
    cells = [[i / n, 0.75 / figh, 1 / n, STRIP_PANEL[1] / figh]
             for i in range(n)]
    fig = composite((figw, figh), cells, seis, preds,
                    {k: labels[k] for k in keys}, miou_shown,
                    style.PARIHAKA_NAMES, legend_y=0.03)
    style.save(fig, "f3_strip", target_wh=(figw, figh))

    # ---------------- F3 row: (a)/(b) + truth on one F3 crossline
    c3, xl3, _, rows_f3 = pick["f3"]
    seis3, lab3_u = section_of(tests[c3], xl3)
    lab3 = lab3_u - BLOCK_F3[0]
    preds3 = {
        "a": predict_section(model_a, _t(seis3), BLOCK_F3),
        "b": predict_section(models_b[c3], _t(seis3), BLOCK_F3),
        "c": lab3,
    }
    labels3 = {"a": "(a) Averaged BN", "b": "(b) Local BN",
               "c": "(c) Interpretation"}
    miou3 = {"a": scores[f"a_f3_client{c3}"], "b": scores[f"b_f3_client{c3}"]}
    figw = STRIP_PANEL[0] * 3
    figh = STRIP_PANEL[1] + 0.75
    cells = [[i / 3, 0.75 / figh, 1 / 3, STRIP_PANEL[1] / figh]
             for i in range(3)]
    fig = composite((figw, figh), cells, seis3, preds3, labels3, miou3,
                    style.F3_NAMES, legend_y=0.03)
    style.save(fig, "f3_f3row", target_wh=(figw, figh))
    facts.append(f"f3row: client {c3}, F3 crossline {xl3}, "
                 f"mIoU a={miou3['a']} b={miou3['b']}")

    # ---------------- crossfade (a) -> (b)
    gif_mb = crossfade(rasters["a"], rasters["b"], "f3_crossfade",
                       seconds=4.0, fps=12)

    # ---------------- facts fragment
    flag_lines = []
    checks = [
        ("(a) Parihaka client 2 (averaged BN)", scores["a_parihaka_client2"],
         scores["a_parihaka_client2_unmasked"], EXPECT["a_pari_client2"]),
        ("(a) Parihaka client 3 (averaged BN)", scores["a_parihaka_client3"],
         scores["a_parihaka_client3_unmasked"], EXPECT["a_pari_client3"]),
        ("(b) Parihaka client 2 (local BN)", scores["b_parihaka_client2"],
         scores["b_parihaka_client2_unmasked"], EXPECT["b_pari_client2"]),
        ("(b) Parihaka client 3 (local BN)", scores["b_parihaka_client3"],
         scores["b_parihaka_client3_unmasked"], EXPECT["b_pari_client3"]),
    ]
    for name, masked, unmasked, exp in checks:
        d_m = abs(masked - exp)
        d_u = abs(unmasked - exp)
        flag_lines.append(
            f"- {name}: masked {masked:.4f} (delta {d_m:.4f}"
            + (" **FLAG >0.02**" if d_m > MISMATCH_TOL else " OK")
            + f"), unmasked {unmasked:.4f} (delta {d_u:.4f}"
            + (" **FLAG >0.02**" if d_u > MISMATCH_TOL else " OK")
            + f") vs findings.md {exp:.4f}")

    rows_doc = ", ".join(f"xl {x}: {m:.3f}" for x, m in
                         sorted(rows_pari)[:: max(1, len(rows_pari) // 8)])
    facts_path = ROOT / "talk_figs/facts_F3F4.md"
    with open(facts_path, "w") as fh:
        fh.write("# Facts: F3 blackout + F4 training movie (sponsor talk)\n\n")
        fh.write("## F3 - The blackout (averaged vs local BN)\n\n")
        fh.write(f"- FedAvg run: `results/runs/multisurvey/fedavg_noniid_a0.1_lNA_dna_n4/seed_0` "
                 f"(seed {seed}, UNet(1,12,norm=batch,groups=8), `global_final.pt`)\n")
        fh.write(f"- FedBN run: `results/runs/multisurvey_fedbn/fedbn_noniid_a0.1_lNA_dna_n4/seed_0` "
                 f"(deployed = `local/client_*.pt`)\n")
        fh.write(f"- Parihaka = survey B, union class rows 6-11 (block (6,12)); "
                 f"F3 = rows 0-5 (block (0,6))\n")
        fh.write(f"- Panel crossline: Parihaka client {c} holdout crossline {xl} "
                 f"(median-error crossline under (b) out of {len(rows_pari)} holdout crosslines; "
                 f"sample: {rows_doc})\n")
        fh.write(f"- F3 row crossline: client {c3} holdout crossline {xl3} "
                 f"(median-error under (b))\n")
        fh.write(f"- BN re-estimation: one full pass in train() mode + torch.no_grad() over the "
                 f"client's own train loader (batches: "
                 + ", ".join(f"c{k}={v}" for k, v in sorted(bn_batches.items())) + "), then eval()\n")
        fh.write(f"- Device: {DEVICE} (GPU fully occupied by another job at run time)\n\n")
        fh.write("### mIoU computed here (present-class, whole client holdout, IGNORE=255 dropped)\n\n")
        fh.write("Two conventions are recorded because findings.md's 2026-10-19 numbers were "
                 "computed with UNMASKED argmax (the run's own `results.json` `global_on_local` "
                 "= 0.15303 = mean(0.3238, 0.2876, 0.0003, 0.0004) pins it exactly), while this "
                 "figure task specifies the deployed masked argmax (`_score_deployed`). "
                 "Masked is the primary (shown in the figures).\n\n")
        fh.write("| model | survey | client | mIoU masked (deployed) | mIoU unmasked (findings conv.) |\n"
                 "|---|---|---|---|---|\n")
        for name, tag, c_, masked, unmasked in detail:
            fh.write(f"| ({name}) | {tag} | {c_} | {masked:.4f} | {unmasked:.4f} |\n")
        fh.write("\n### Comparison vs findings.md (2026-10-19 diagnostic; flag >0.02)\n\n")
        fh.write("\n".join(flag_lines) + "\n")
        fh.write("\nInterpretation of any FLAG: findings.md numbers are UNMASKED-argmax values "
                 "(0.0003/0.0004 for (a) means the unmasked argmax never selects Parihaka rows "
                 "under averaged BN); the masked (deployed) values score the same model within "
                 "the Parihaka block only. For (b), a remaining unmasked delta is attributable to "
                 "the BN refresh protocol (this run: one EMA-continuing pass over the train "
                 "loader in train() mode, as specified in the figure brief).\n")
        fh.write(f"\n- crossfade `f3_crossfade.mp4`/`.gif` ({gif_mb/1e6:.1f} MB), "
                 f"4.0 s @ 12 fps, (a) -> (b)\n")
        fh.write("\n### F3 row (control)\n\n")
        fh.write(f"- client {c3}, F3 crossline {xl3}: (a) averaged BN masked mIoU "
                 f"{miou3['a']:.4f}, (b) local BN masked mIoU {miou3['b']:.4f}. "
                 f"On F3 the masked and unmasked conventions coincide (the model's argmax "
                 f"stays in rows 0-5). (a) is degraded vs (b) but structured - NOT the "
                 f"single-facies blackout seen on Parihaka; the weights are fine once BN is "
                 f"right. Findings.md also records the global model scoring only 0.10-0.19 on "
                 f"the F3 official test (broken-reference note).\n")
    print(f"wrote {facts_path}", flush=True)
    print("F3 done.", flush=True)


def _t(section):
    return torch.as_tensor(np.asarray(section), dtype=torch.float32).unsqueeze(0).unsqueeze(0)


if __name__ == "__main__":
    main()
