"""F4 sponsor-talk figure: training movie - facies appearing over 20 rounds.

Frames = predictions of the round-k global model of the seismic_f3 FedAvg
run (norm=batch, N=3, noniid a0.1), each round state loaded as-is (its own
averaged BN running stats), eval mode, overlaid on the seismic section of one
fixed site-holdout crossline (the median-error crossline, not the best one).

The round checkpoints live on the HDD (`received/round_001..020.pt`); the
run config comes from the local mirror's `run_meta.json` (the HDD copy holds
only received/ and view/).

Run:  cd <repo> && fl-seismic/bin/python talk_figs/f4_training_movie.py
"""
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
from fedseismic.eval.metrics import _logits, _present_mean
from fedseismic.experiment import build_federated_run
from fedseismic.models import UNet
from fedseismic.privacy.log import load_meta

LOCAL_RUN = ROOT / "results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0"
RECEIVED = Path("/mnt/storage/results/runs/seismic_f3/"
                "fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0/received")
ROUNDS = list(range(1, 21))
DEVICE = "cpu"   # GPU occupied by another job at run time
FPS = 11          # 10-12 fps per the slide spec
HOLD_LAST_S = 2.0
DPI_RASTER = 150
PAD = 0.3          # style.save pads 0.15 in per side; compensate below


def canvas_wh(aspect, height_out):
    """figure size whose saved output (figure + style.save's pad) has `aspect`."""
    return (aspect * height_out - PAD, height_out - PAD)


FRAME_WH = canvas_wh(style.ASPECT["f4movie"], 4.0)   # saved stills = 1.83:1
MOVIE_WH = (style.ASPECT["f4movie"] * 4.0, 4.0)      # animation canvas = 1.83:1 (no save pad)
GRID_WH = (10.98, 6.0)                               # 2x2 grid of 1.83:1 panels


def build_model(num_classes):
    return UNet(1, num_classes, bilinear=False, norm="batch", groups=8)


def predict(model, image):
    with torch.no_grad():
        return _logits(model(image.to(DEVICE, dtype=torch.float))).argmax(1)[0].cpu().numpy()


def per_sample_miou(model, loader, num_classes):
    ds = loader.dataset
    rows = []
    with torch.no_grad():
        for images, labels, pos in loader:
            preds = _logits(model(images.to(DEVICE, dtype=torch.float))).argmax(1).cpu().numpy()
            for i in range(images.size(0)):
                yt = np.asarray(labels[i]).ravel()
                yp = preds[i].ravel()
                keep = yt != 255  # 255 == fedseismic.eval.metrics.IGNORE
                yt, yp = yt[keep], yp[keep]
                per = jaccard_score(yt, yp, labels=list(range(num_classes)),
                                    average=None, zero_division=0)
                rows.append((int(ds.indices[int(pos[i])]),
                             float(_present_mean(per, yt, num_classes))))
    return rows


def fig_raster(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


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


def render_frame(seis, pred, round_k, n=6, cell=None, fig_wh=None):
    """Frame on the 1.83:1 slide canvas: seismic at TRUE aspect (letterboxed),
    facies overlay + thin boundary contour, small 'Round k' corner label."""
    fig_wh = fig_wh or FRAME_WH
    fig = plt.figure(figsize=fig_wh, dpi=DPI_RASTER)
    pin_canvas(fig)
    ar = seis.shape[1] / seis.shape[0]
    ax = fig.add_axes(image_rect(fig_wh, cell or [0, 0, 1, 1], ar))
    style.show_seismic(ax, seis)
    style.overlay_facies(ax, pred, n=n, alpha=0.45, contour=True)
    style.label_panel(ax, f"Round {round_k}", size=14)
    return fig


def save_movie(frames, stem):
    """MP4 (H.264 veryfast) + palette GIF + final-frame PNG from RGB frames."""
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    h = frames[0].shape[0] // 2 * 2
    w = frames[0].shape[1] // 2 * 2
    frames = [f[:h, :w] for f in frames]
    fig = plt.figure(figsize=(w / DPI_RASTER, h / DPI_RASTER))
    ax = fig.add_axes([0, 0, 1, 1])
    im = ax.imshow(frames[0], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])

    def update(k):
        im.set_data(frames[k])
        return [im]

    anim = FuncAnimation(fig, update, frames=len(frames), blit=True)
    out = Path(style.OUT)
    out.mkdir(parents=True, exist_ok=True)
    mp4 = out / f"{stem}.mp4"
    anim.save(str(mp4), writer=FFMpegWriter(
        fps=FPS, codec="h264", bitrate=2400,
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

    tmp = Path("/tmp/opencode/f4movie")
    tmp.mkdir(parents=True, exist_ok=True)
    for p in tmp.glob("f_*.png"):
        p.unlink()
    for i, f in enumerate(frames):
        plt.imsave(tmp / f"f_{i:03d}.png", f)
    pal = tmp / "palette.png"
    gif = out / f"{stem}.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", str(tmp / "f_%03d.png"), "-vf",
                    "palettegen=stats_mode=diff", str(pal)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", str(tmp / "f_%03d.png"), "-i", str(pal), "-lavfi",
                    "paletteuse=dither=bayer:bayer_scale=3", str(gif)], check=True)
    print(f"  wrote {mp4}  {gif} ({gif.stat().st_size/1e6:.1f} MB)  "
          f"final {out}/{stem}_final.png")
    return gif.stat().st_size


def main():
    torch.set_num_threads(os.cpu_count() or 8)

    # ---- inputs present?
    missing = [k for k in ROUNDS if not (RECEIVED / f"round_{k:03d}.pt").is_file()]
    if missing:
        print(f"SKIP F4: missing received/round_*.pt for rounds {missing} "
              f"under {RECEIVED}")
        return
    print(f"found all {len(ROUNDS)} round checkpoints under {RECEIVED}", flush=True)

    meta = load_meta(LOCAL_RUN)
    cfg = RunConfig.from_mapping(meta["config"])
    seed = int(meta.get("seed", cfg.seed))
    print(f"building federated run (seed {seed}) ...", flush=True)
    data = build_federated_run(cfg, seed)
    num_classes = int(cfg.num_classes)

    def load_round(k):
        state = torch.load(RECEIVED / f"round_{k:03d}.pt", map_location="cpu",
                           weights_only=True)
        m = build_model(num_classes)
        m.load_state_dict(state)
        return m.to(DEVICE).eval()

    # ---- fixed display crossline: median per-crossline mIoU under the LAST
    # round across all site holdouts (representative, not the best one)
    model_last = load_round(ROUNDS[-1])
    rows = []
    for c, loader in enumerate(data.client_tests):
        if loader is None or not len(loader.dataset):
            continue
        for xl, miou in per_sample_miou(model_last, loader, num_classes):
            rows.append((c, xl, miou))
    rows_sorted = sorted(rows, key=lambda r: r[2])
    med_c, med_xl, med_miou = rows_sorted[len(rows_sorted) // 2]
    q1 = rows_sorted[len(rows_sorted) // 4][2]
    q3 = rows_sorted[3 * len(rows_sorted) // 4][2]
    print(f"median-error crossline: client {med_c} xl {med_xl} "
          f"mIoU {med_miou:.4f} (Q1 {q1:.3f} Q3 {q3:.3f}, n={len(rows)})", flush=True)

    ds = data.client_tests[med_c].dataset
    seis = np.asarray(ds.seismic[:, med_xl, :]).T
    truth = np.asarray(ds.label[:, med_xl, :]).T

    # ---- frames: round-k global, eval mode, its own averaged BN (as saved)
    frames = []
    round_pred = {}
    for k in ROUNDS:
        model = load_round(k)
        pred = predict(model, _t(seis))
        round_pred[k] = pred
        fig = render_frame(seis, pred, k, n=num_classes, fig_wh=MOVIE_WH)
        frames.append(fig_raster(fig))
        plt.close(fig)
        if k in (1, 5, 10, 20):
            style.save(render_frame(seis, pred, k, n=num_classes),
                       f"f4_round{k:02d}", target_wh=FRAME_WH)
        print(f"  frame round {k} done", flush=True)

    # hold the last frame 2 s
    frames += [frames[-1]] * int(round(HOLD_LAST_S * FPS))
    gif_mb = save_movie(frames, "f4_training_movie")

    # ---- 4-panel grid: rounds 1, 5, 10, 20 (2x2, 1.83:1 cells, TRUE aspect)
    picks = [1, 5, 10, 20]
    fig = plt.figure(figsize=GRID_WH)
    pin_canvas(fig)
    ar = seis.shape[1] / seis.shape[0]
    for i, k in enumerate(picks):
        cell = [(i % 2) / 2, 0.5 - (i // 2) / 2, 0.5, 0.5]
        ax = fig.add_axes(image_rect(GRID_WH, cell, ar))
        style.show_seismic(ax, seis)
        style.overlay_facies(ax, round_pred[k], n=num_classes, alpha=0.45,
                             contour=True)
        style.label_panel(ax, f"Round {k}", size=14)
    style.save(fig, "f4_rounds_grid", target_wh=GRID_WH)

    # ---- facts fragment (appended; F3 section written by f3_blackout.py)
    median_row = ", ".join(f"c{c} xl{xl}: {m:.3f}" for c, xl, m in sorted(rows)[:: max(1, len(rows) // 10)])
    facts_path = ROOT / "talk_figs/facts_F3F4.md"
    with open(facts_path, "a") as fh:
        fh.write("\n## F4 - Training movie (20 rounds, facies appearing)\n\n")
        fh.write(f"- run: `results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0` "
                 f"(config from local `run_meta.json`, seed {seed}, UNet(1,{num_classes},norm=batch,groups=8))\n")
        fh.write(f"- round checkpoints: `{RECEIVED}/round_001..020.pt` (all 20 present) — "
                 f"loaded as-is in eval mode = each round's own averaged BN running stats\n")
        fh.write(f"- fixed crossline: client {med_c} site holdout, crossline {med_xl} — "
                 f"median per-crossline present-class mIoU under round 20 over all "
                 f"{len(rows)} holdout crosslines (mIoU {med_miou:.4f}; Q1 {q1:.4f}, Q3 {q3:.4f}; "
                 f"sample: {median_row})\n")
        fh.write(f"- movie: 20 frames (round 1..20) + {int(round(HOLD_LAST_S * FPS))} hold frames "
                 f"= {len(frames)/FPS:.1f} s @ {FPS} fps, H.264 -preset veryfast; "
                 f"GIF {gif_mb/1e6:.1f} MB (<=15 MB)\n")
        fh.write(f"- artifacts: `f4_training_movie.mp4`/`.gif`/`_final.png`, `f4_rounds_grid.png`, "
                 f"round stills `f4_round01/05/10/20.png` (PNG 300dpi + PDF + SVG via style.save)\n")
        fh.write(f"- device: {DEVICE} (GPU fully occupied by another job at run time)\n")
    print(f"appended F4 facts to {facts_path}", flush=True)
    print("F4 done.", flush=True)


def _t(section):
    return torch.as_tensor(np.asarray(section), dtype=torch.float32).unsqueeze(0).unsqueeze(0)


if __name__ == "__main__":
    main()
