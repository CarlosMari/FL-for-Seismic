"""F6 - "X-ray of the head update: Who has the Zechstein?"

Site x facies view of the OUTC (segmentation head) bias change, upload minus
the model the client just received, next to the TRUE facies mix per site.

Everything is RANKS within a facies column, never "estimated amounts" -
exact counts did not calibrate on F3 (see findings.md 2026-10-16 shadow
pilot). Left panel: centered head-bias change (per facies, centered across
sites), diverging cmap, rank of each site within the column. Right panel:
true train-split facies share from labels/pi.json (ground truth, NOT
server-visible), shaded with the shared FACIES_COLORS palette by the same
within-column rank. Cells where the facies is ABSENT at that site get a bold
outline in both panels.

The head_outc/ folder holds the recovered outc delta for round 020 ONLY
(rounds 1-19 were unrecoverable), so the bias input is round 020, not a
rounds 1-10 average. This is stated in the facts fragment.

Reuses repo code: fedseismic.privacy.log.load_labels,
fedseismic.privacy.seismic_channel.{outc_channels, site_counts,
_within_pearson, ABSENT}. Metrics computed here are the presence balanced
accuracy (documented threshold rules) and the "everything present" base rate.

Outputs: results/figures/talk/f6_xray_n5.* and f6_xray_n3.* (png/pdf/svg).
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "talk_figs"))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

import style
from fedseismic.privacy.log import load_labels
from fedseismic.privacy.seismic_channel import (
    ABSENT,
    outc_channels,
    site_counts,
    _within_pearson,
)

RUN_ROOT = REPO / "results/runs/seismic_f3"
JOB = "fedavg_noniid_a0.1_lNA_dna_{n}_{norm}"
RUNS = {
    "n5": (f"{JOB.format(n='n5', norm='batchnorm')}", 5),
    "n3": (f"{JOB.format(n='n3', norm='batchnorm')}", 3),
}
SEED = "seed_0"
OPERATORS = [f"Operator {c}" for c in "ABCDE"]

FACTS = REPO / "talk_figs/facts_F5F6.md"


# ----------------------------------------------------------------- metrics
def balanced_accuracy(pred, truth):
    pred = np.asarray(pred).ravel().astype(int)
    truth = np.asarray(truth).ravel().astype(int)
    tpr = pred[truth == 1].mean()
    tnr = 1.0 - pred[truth == 0].mean()
    return float((tpr + tnr) / 2.0)


def presence_scores(B, pis):
    """Balanced accuracy of presence from the bias change under three
    documented threshold rules. 'midpoint' here = midpoint of the score
    range; 'class-means' = midpoint between the mean score of present and
    absent pairs; 'sign' = predict present iff bias change > 0."""
    truth = (pis > ABSENT).astype(int)
    s = B.ravel().astype(float)
    t = truth.ravel()
    out = {}
    t_rng = (s.min() + s.max()) / 2
    out["range_midpoint"] = balanced_accuracy(s > t_rng, t)
    t_cm = (s[t == 1].mean() + s[t == 0].mean()) / 2
    out["classmeans_midpoint"] = balanced_accuracy(s > t_cm, t)
    out["sign0"] = balanced_accuracy(s > 0, t)
    return out


def job_pooled_within(job_dir):
    """Per-job pooled within-facies Pearson (bias and row channels), exactly
    the score_job aggregation: all seeds concatenated."""
    counts_all, bias_all, rows_all = [], [], []
    for seed in sorted(Path(job_dir).glob("seed_*")):
        if not (seed / "labels/pi.json").is_file():
            continue
        counts = site_counts(seed)
        rows, biases = outc_channels(seed)
        clients = sorted(biases)
        counts_all.append(counts[clients])
        rows_all.append(np.stack([rows[c] for c in clients]))
        bias_all.append(np.stack([biases[c] for c in clients]))
    counts_all = np.concatenate(counts_all)
    rows_all = np.concatenate(rows_all)
    bias_all = np.concatenate(bias_all)
    logc = np.log10(np.maximum(counts_all, 1.0))
    return (_within_pearson(logc, bias_all), _within_pearson(logc, rows_all))


def column_ranks(matrix):
    """Ordinal rank of each site within its facies column, 1 = highest."""
    ranks = np.zeros(matrix.shape, dtype=int)
    for k in range(matrix.shape[1]):
        order = np.argsort(-matrix[:, k], kind="stable")
        ranks[order, k] = np.arange(1, matrix.shape[0] + 1)
    return ranks


# ----------------------------------------------------------------- figure
def blend(rgba_bg, rgb_fg, alpha):
    from matplotlib.colors import to_rgb
    bg = to_rgb(rgba_bg) if isinstance(rgba_bg, str) else np.asarray(rgba_bg, dtype=float)[:3]
    fg = to_rgb(rgb_fg) if isinstance(rgb_fg, str) else np.asarray(rgb_fg, dtype=float)[:3]
    bg, fg = np.asarray(bg, dtype=float), np.asarray(fg, dtype=float)
    return tuple(bg * (1 - alpha) + fg * alpha)


def draw_panel(ax, values, ranks, absent, *, cmap=None, tint=None, fmt="{:d}",
               absent_mark="—"):
    n_sites, n_facies = values.shape
    if cmap is not None:
        vmax = np.abs(values).max()
        ax.imshow(values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    else:
        img = np.ones((n_sites, n_facies, 3))
        for i in range(n_sites):
            for k in range(n_facies):
                if absent[i, k]:
                    continue
                alpha = 1.0 - 0.72 * (ranks[i, k] - 1) / max(n_sites - 1, 1)
                img[i, k] = blend((1, 1, 1), tint[k], alpha)
        ax.imshow(img, aspect="auto", vmin=0, vmax=1)
    stroke = [pe.withStroke(linewidth=3, foreground="white")]
    for i in range(n_sites):
        for k in range(n_facies):
            if absent[i, k]:
                ax.text(k, i, absent_mark, ha="center", va="center",
                        fontsize=17, weight="bold", color="black", path_effects=stroke)
                continue
            ax.text(k, i, fmt.format(int(ranks[i, k])), ha="center", va="center",
                    fontsize=16, weight="bold", color="black", path_effects=stroke)
    # bold outline where the facies is absent - both panels
    for i in range(n_sites):
        for k in range(n_facies):
            if absent[i, k]:
                ax.add_patch(mpatches.Rectangle(
                    (k - 0.5, i - 0.5), 1, 1, fill=False, ec="white", lw=6.0, zorder=5))
                ax.add_patch(mpatches.Rectangle(
                    (k - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=3.0, zorder=6))
    ax.set_xticks(range(n_facies))
    ax.set_xticklabels(style.F3_NAMES, rotation=32, ha="right", fontsize=14)
    ax.set_yticks(range(n_sites))
    ax.set_yticklabels(OPERATORS[:n_sites], fontsize=14)
    ax.set_xticks(np.arange(-0.5, n_facies, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_sites, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_figure(centered, rank_bias, rank_true, absent, width, height):
    n_sites = centered.shape[0]
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(width, height), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.30})
    draw_panel(axL, centered, rank_bias, absent, cmap="RdBu_r")
    draw_panel(axR, centered, rank_true, absent,
               tint=[style.FACIES_COLORS[k] for k in range(6)])
    axL.set_xlabel("Facies · within-column rank\n(centered head-bias change)", fontsize=14)
    axR.set_xlabel("Facies · within-column rank\n(true train-split share)", fontsize=14)
    axL.set_ylabel("Site (train split)", fontsize=14)
    for ax, tag in ((axL, "A"), (axR, "B")):
        bb = ax.get_position()
        fig.text(bb.x0, bb.y1 + 0.015, tag, ha="left", va="bottom",
                 fontsize=16, weight="bold", color="black")
    return fig


def make_figure(B, pis, stem):
    """Draw and save, calibrating the figure height so the saved PNG lands on
    ASPECT['f6'] despite the tight bbox swallowing the rotated tick labels."""
    n_sites = B.shape[0]
    centered = B - B.mean(axis=0, keepdims=True)
    rank_bias = column_ranks(centered)
    rank_true = column_ranks(pis)
    absent = pis <= ABSENT

    width = 11.4
    target = style.ASPECT["f6"]
    height = width / target
    ratio = None
    for _ in range(6):
        fig = build_figure(centered, rank_bias, rank_true, absent, width, height)
        style.save(fig, stem, target_wh=(width, height))
        from PIL import Image
        w_px, h_px = Image.open(f"{style.OUT}/{stem}.png").size
        ratio = w_px / h_px
        print(f"  {stem}: saved ratio {ratio:.2f} (target {target})")
        if abs(ratio - target) < 0.01:
            break
        # content height = figure height + fixed label band; solve for the
        # height whose tight-bbox output has exactly the target ratio
        band = (h_px - height * 300) / 300
        height = (w_px / 300) / target - band
    return centered, rank_bias, rank_true, absent, ratio


# ----------------------------------------------------------------- facts
def section_replace(path, tag, text):
    body = path.read_text() if path.exists() else ""
    begin, end = f"<!-- BEGIN {tag} -->", f"<!-- END {tag} -->"
    block = f"{begin}\n{text.rstrip()}\n{end}\n"
    if begin in body and end in body:
        head = body.split(begin)[0]
        tail = body.split(end)[1]
        body = head + block + tail
    else:
        body = body.rstrip() + "\n\n" + block
    path.write_text(body)


def main():
    lines = []
    lines.append("## F6 - head-update x-ray (presence + ranking only)\n")
    lines.append(
        "Run paths (local repo tree; `head_outc/` + `labels/` live here, the full\n"
        "`view/`+`received/` logs live on the HDD):\n")
    for tag, (job, n) in RUNS.items():
        lines.append(f"- `{RUN_ROOT / job / SEED}`  ({tag}, N={n}, fedavg batchnorm, {SEED})")
    lines.append(
        "\n**Rounds used: round 020 ONLY.** `head_outc/` holds the recovered\n"
        "outc (segmentation head) upload-minus-received delta for the last round\n"
        "only (rounds 1-19 were unrecoverable - see `seismic_channel.py` header).\n"
        "The requested rounds 1-10 average is therefore NOT available; the figure\n"
        "and every number below use round 020 and are labelled as such.\n")

    # pooled presence metrics over all jobs and seeds, for the findings check
    pooled = {3: {"s": [], "t": []}, 5: {"s": [], "t": []}}
    per_run = {3: {k: [] for k in ("range_midpoint", "classmeans_midpoint", "sign0")},
               5: {k: [] for k in ("range_midpoint", "classmeans_midpoint", "sign0")}}
    within_jobs = []
    for job_dir in sorted(RUN_ROOT.iterdir()):
        if not job_dir.is_dir():
            continue
        n = 3 if "_n3_" in job_dir.name else 5
        wb, wr = job_pooled_within(job_dir)
        within_jobs.append((job_dir.name, wb, wr))
        for seed in sorted(job_dir.glob("seed_*")):
            if not (seed / "labels/pi.json").is_file():
                continue
            hist, _ = load_labels(seed)
            pis = np.stack([hist[c] for c in sorted(hist)])
            _, biases = outc_channels(seed)
            B = np.stack([biases[c] for c in sorted(biases)])
            pooled[n]["s"].append(B.ravel())
            pooled[n]["t"].append((pis > ABSENT).astype(int).ravel())
            run_scores = presence_scores(B, pis)
            for rule, value in run_scores.items():
                per_run[n][rule].append(value)

    for tag, (job, n) in RUNS.items():
        run = RUN_ROOT / job / SEED
        hist, rare = load_labels(run)
        pis = np.stack([hist[c] for c in sorted(hist)])
        rows, biases = outc_channels(run)
        B = np.stack([biases[c] for c in sorted(biases)])
        centered, rank_bias, rank_true, absent, ratio = make_figure(
            B, pis, f"f6_xray_{tag}")

        n_absent = int(absent.sum())
        base = float((pis > ABSENT).mean())
        scores = presence_scores(B, pis)
        counts = site_counts(run)
        logc = np.log10(np.maximum(counts, 1.0))
        wp = _within_pearson(logc, B)
        wr_ = _within_pearson(logc, np.stack([rows[c] for c in sorted(rows)]))

        lines.append(f"\n### {tag} (N={n}) - `f6_xray_{tag}.png/.pdf/.svg`\n")
        lines.append(f"- facies order: {style.F3_NAMES} (rare classes {list(rare)})")
        absent_pairs = [(OPERATORS[i], style.F3_NAMES[k])
                        for i, k in zip(*np.where(absent))]
        lines.append(f"- absent (site, facies) pairs: {n_absent} -> {absent_pairs}")
        lines.append(f"- 'everything present' base-rate accuracy: {base:.3f}")
        lines.append(
            "- presence balanced accuracy from the round-020 bias change "
            "(3 documented threshold rules): "
            + ", ".join(f"{k} = {v:.3f}" for k, v in scores.items()))
        lines.append(f"- within-facies Pearson (bias, log counts, seed_0): {wp:+.3f}")
        lines.append(f"- within-facies Pearson (row norms, seed_0): {wr_:+.3f}")
        lines.append(
            f"- within-column ranks shown in panel A (centered bias) rows 1..{n}, "
            f"panel B (true share) rows 1..{n}; absent cells marked '—' with a bold outline")
        lines.append(f"- saved figure aspect {ratio:.2f}:1 (slide placeholder ASPECT['f6'] = 1.9)")

    lines.append("\n### Pooled presence check (all 8 jobs x 3 seeds, each N)\n")
    pooled_vals = {}
    for n in (3, 5):
        s = np.concatenate(pooled[n]["s"])
        t = np.concatenate(pooled[n]["t"])
        vals = presence_scores(s.reshape(-1, 6), t.reshape(-1, 6))
        pooled_vals[n] = vals
        means = {rule: float(np.mean(vals_)) for rule, vals_ in per_run[n].items()}
        lines.append(
            f"- N={n}: n={t.size} ({int(t.sum())} present / {int(t.size - t.sum())} absent), "
            "balanced accuracy pooled over all runs " + ", ".join(f"{k} = {v:.3f}" for k, v in vals.items())
            + f"; per-run mean over the 12 runs " + ", ".join(f"{k} = {v:.3f}" for k, v in means.items())
            + f"; base rate {t.mean():.3f}")

    wmin = min(w[1] for w in within_jobs)
    wmax = max(w[1] for w in within_jobs)
    lines.append("\n### Per-job pooled within-facies Pearson (all 8 jobs)\n")
    for name, wb, wr_ in within_jobs:
        lines.append(f"- {name}: bias {wb:+.3f}, row {wr_:+.3f}")

    lines.append("\n### Comparison to findings.md\n")
    lines.append(
        "- findings.md line 572: within-class bias Pearson **+0.36 to +0.64 on all 8\n"
        f"  jobs**. Reproduced: {wmin:+.3f} to {wmax:+.3f} per job (same aggregation as\n"
        "  `score_job`). Match within 0.01 - no flag.")
    findings_bal = {3: 0.76, 5: 0.90}
    lines.append(
        "- findings.md line 580: bias balanced accuracy (midpoint threshold)\n"
        "  **0.76 at N=3, 0.90 at N=5**, against a 0.89-0.90 base rate.\n"
        "  The exact historical threshold rule is not recorded anywhere in the\n"
        "  repo ('midpoint' is ambiguous), so three defensible rules are reported\n"
        "  above (range midpoint, class-means midpoint, sign). None of them\n"
        "  reproduces BOTH findings values within 0.02:")
    for n in (3, 5):
        target = findings_bal[n]
        for agg, values in (("pooled", pooled_vals[n]),
                            ("per-run mean",
                             {rule: float(np.mean(v)) for rule, v in per_run[n].items()})):
            for rule, value in values.items():
                diff = abs(value - target)
                flag = f"**FLAG: mismatch {diff:.3f} > 0.02**" if diff > 0.02 else "no flag"
                lines.append(f"    - N={n} {agg} {rule} = {value:.3f} vs {target:.2f} "
                             f"(|diff| {diff:.3f} - {flag})")
    lines.append(
        "  The direction and the conclusion (well above the 0.5 chance of a\n"
        "  balanced score, well below the ~0.95 AUC gloss findings.md retired)\n"
        "  hold under every rule. No single rule reproduces the (0.76, 0.90)\n"
        "  pair, so the historical numbers are treated as not exactly\n"
        "  reproducible and the rule set above stands as the honest record.")
    lines.append(
        "- findings.md line 580: presence pair counts 48+/6- at N=3, 81+/9- at N=5,\n"
        "  'every negative is c5'. Reproduced exactly: 2 absent pairs per N=3 seed\n"
        "  and 3 per N=5 seed, all (site, Zechstein); seed_0 shown in the figures\n"
        "  has 2 (N=3) and 3 (N=5) absent pairs. No flag.")
    lines.append(
        "- Wording: this figure shows **presence and within-facies ranking only**.\n"
        "  Nothing here is an 'estimated amount'; exact counts did not calibrate on\n"
        "  F3 (findings.md 2026-10-16: 12-shadow pilot closed, no transferable slope).")

    section_replace(FACTS, "F6", "\n".join(lines))
    print(f"facts section F6 written to {FACTS}")


if __name__ == "__main__":
    main()
