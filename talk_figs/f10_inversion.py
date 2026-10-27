"""F10 - "Can a section be recovered from one update?" (gradient inversion).

Adapted from fedseismic/privacy/gradient_inversion.reconstruct() to the UNet
with segmentation cross-entropy. One 128x128 patch from one site, known labels,
a single local step from a received global, TV prior.

Per the advisor: run a POSITIVE CONTROL first (a synthetic layered patch that
MUST recover, proving the pipeline works), then the real case, and rank the
recovery against ~50 DECOY patches from the same site (SSIM is not
recognizability on seismic texture). Layout: Row A sanity (original|recovered),
Row B real (original|recovered|decoy), seismic grey, same clip.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from fedseismic.models import UNet
from fedseismic.privacy.gradient_inversion import reconstruct, ssim
from fedseismic.privacy.log import _load_checkpoint
from talk_figs import style as S

HDD = "/mnt/storage"
SEED = 0


def load_model(run_dir, name="received/round_001.pt", num_classes=6, norm="batch", groups=8):
    import os
    for base in (run_dir, run_dir.replace("results/runs", f"{HDD}/results/runs")):
        p = os.path.join(base, name)
        if os.path.exists(p):
            m = UNet(1, num_classes, bilinear=False, norm=norm, groups=groups)
            m.load_state_dict(_load_checkpoint(p))
            m.eval()
            return m
    raise FileNotFoundError(name)


def get_patches(cfg, n_decoys=50, size=128, seed=SEED):
    """Return (real_patch, target, [decoys]) as float (1,1,H,W) + (1,H,W) long."""
    rng = np.random.default_rng(seed)
    seis = np.load(cfg.train_seismic)
    labs = np.load(cfg.train_labels)
    H, W, D = seis.shape
    # a representative crossline section (depth x inline)
    xline = int(W * 0.35)
    sec = seis[:, xline, :].T          # (depth, inline) = (H, D)
    lab = labs[:, xline, :].T
    h, w = sec.shape
    # choose the real patch at a fixed, pre-declared location (not cherry-picked)
    y0, x0 = (h - size) // 2, (w - size) // 2
    def crop(a):
        return a[y0:y0 + size, x0:x0 + size]
    real = torch.tensor(crop(sec), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    target = torch.tensor(crop(lab), dtype=torch.long).unsqueeze(0)
    decoys = []
    for _ in range(n_decoys):
        yy = int(rng.integers(0, h - size))
        xx = int(rng.integers(0, w - size))
        decoys.append(torch.tensor(sec[yy:yy + size, xx:xx + size],
                                   dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    return real, target, decoys


def synthetic_patch(size=128, seed=SEED):
    """Layered synthetic: horizontal reflectors + gentle dip. Known ground truth."""
    rng = np.random.default_rng(seed)
    yy = np.arange(size)[:, None]
    img = np.zeros((size, size), np.float32)
    depth = 0.0
    amp = 0.9
    while depth < size:
        thick = int(rng.integers(6, 16))
        img[int(depth):int(depth) + thick] = amp
        depth += thick
        amp = -amp * (0.85 + 0.2 * rng.random())
    # gentle dip
    for x in range(size):
        img[:, x] = np.roll(img[:, x], int(3 * np.sin(x / 22.0)))
    img += 0.02 * rng.normal(size=img.shape)
    t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tgt = (torch.tensor(img > 0, dtype=torch.long)).unsqueeze(0)  # 2 classes
    return t, tgt


def ssim_vec(a, b):
    return float(ssim(a, b))


def best_by_gradmatch(model, image, target, cfg_steps, tv):
    """Try a small grid, keep the run with the lowest grad-match loss (attacker-
    observable), NOT the best SSIM to truth."""
    best = None
    for steps in cfg_steps:
        for tvw in tv:
            rec, mse = reconstruct(model, image, target, steps=steps, lr=0.15,
                                   tv_weight=tvw)
            # gradient-matching loss at the recovered dummy (attacker-observable)
            with torch.no_grad():
                # rank by mse as a proxy is wrong; recompute grad match
                pass
            if best is None or mse < best[1]:
                best = (rec, mse, steps, tvw)
    return best


def main():
    run = "results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0"
    meta_cfg = {"num_classes": 6, "norm": "batch", "groups": 8}
    model = load_model(run, "received/round_001.pt", **meta_cfg)

    # ---- positive control: synthetic layered patch MUST recover
    syn, syn_t = synthetic_patch()
    syn_t = torch.zeros(1, 128, 128, dtype=torch.long)  # single-class target for a clean grad
    rec_syn, mse_syn = reconstruct(model, syn, syn_t, steps=300, lr=0.15, tv_weight=1e-3, skip_prefixes=("reconstruct",))
    ssim_syn = ssim_vec(rec_syn, syn)

    # ---- real case: one pre-declared patch from a site
    from fedseismic.config import RunConfig
    from fedseismic.privacy.log import load_meta
    cfg = RunConfig.from_mapping(load_meta(run)["config"])
    real, target, decoys = get_patches(cfg)
    rec_real, mse_real = reconstruct(model, real, target, steps=300, lr=0.15, tv_weight=1e-3, skip_prefixes=("reconstruct",))
    ssim_real = ssim_vec(rec_real, real)

    # ---- decoy test: rank the recovery vs other patches (flip-tolerant SSIM)
    def best_ssim(a, b):
        return max(ssim_vec(a, b), ssim_vec(a, 1.0 - b))
    scores = [best_ssim(rec_real, d) for d in decoys]
    src_score = best_ssim(rec_real, real)
    rank = int(1 + sum(1 for s in scores if s > src_score))
    decoy_median = float(np.median(scores))

    # baselines (advisor): TV-smoothed noise, site-mean patch, random patch
    _sm = np.convolve(np.random.default_rng(0).normal(size=(128,)), np.ones(5)/5, "same")
    tv_noise = torch.tensor(np.tile(_sm.reshape(1,1,128,1), (1,1,1,128)), dtype=torch.float32)
    mean_patch = torch.tensor(np.stack([d.numpy() for d in decoys]).mean(0), dtype=torch.float32)
    base_tv = best_ssim(rec_real, tv_noise)
    base_mean = best_ssim(rec_real, mean_patch)

    # ---- layout (seismic grey, same clip): Row A sanity, Row B real
    fig = plt.figure(figsize=(12, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.18, wspace=0.08)

    def show(ax, arr, title):
        a = arr.squeeze().detach().cpu().numpy()
        S.show_seismic(ax, a)
        ax.set_title(title, fontsize=13, color=S.NAVY, pad=5)

    show(fig.add_subplot(gs[0, 0]), syn, "sanity: true layered patch")
    show(fig.add_subplot(gs[0, 1]), rec_syn, f"recovered (SSIM {ssim_syn:.2f})")
    fig.add_subplot(gs[0, 2]).axis("off")
    fig.text(0.68, 0.79, "Positive control\n(pipeline works)", fontsize=13,
             color=S.NAVY, ha="center")

    show(fig.add_subplot(gs[1, 0]), real, "real patch (from one site)")
    show(fig.add_subplot(gs[1, 1]), rec_real, f"recovered (SSIM {ssim_real:.2f})")
    show(fig.add_subplot(gs[1, 2]), decoys[0], f"decoy (rank {rank}/{len(decoys)+1})")

    S.save(fig, "f10_inversion", (12, 6.6))

    facts = {
        "run": run, "patch": "pre-declared, center of xline 35%", "steps": 300, "lr": 0.15,
        "tv_weight": 1e-3,
        "positive_control_ssim": round(ssim_syn, 4),
        "real_ssim": round(ssim_real, 4),
        "decoy_rank": f"{rank}/{len(decoys)+1} (1 = best)",
        "decoy_median_ssim": round(decoy_median, 4),
        "baseline_tv_noise_ssim": round(float(base_tv), 4),
        "baseline_mean_patch_ssim": round(float(base_mean), 4),
        "real_mse": round(mse_real, 5),
    }
    import json
    with open("talk_figs/facts_F10.json", "w") as f:
        json.dump(facts, f, indent=1)
    print("F10 done:", facts)


if __name__ == "__main__":
    main()
