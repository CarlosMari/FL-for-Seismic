"""Few-silo encrypted sums, simulated in plaintext.

The protocol claim (handoff section 4.1) is folklore and gets one paragraph:
with full, stable participation the best any protocol achieves against peers
is an encrypted sum (Elkordy et al., PoPETs 2023; Lam et al., ICML 2021; the
POSEIDON N-of-N collective-CKKS design, NDSS 2021). This module measures what
that sum still reveals. No crypto runs here: the sum is the mean of the
members' saved per-participation classifier deltas.

Observers, per run (one N, one alpha, one method, one seed):
- server-individual: each member's own mean head delta, scored as in howmuch.
- server-sum: the round-averaged mean delta of the cohort. One feature vector
  per run. Attribution asks: whose counts does it name? Every member is
  scored against it; the reported number is the mean over members.
- peer-sum: member j sees the cohort sum and subtracts its own update, then
  is scored on the residual as an estimate of the *mean* of the other
  members' updates. The count line predicts single-client-scale counts, so
  the target is the mean of the others' counts. Reported as one pooled R2
  per seed across members.

Attackers reuse the howmuch shadow fits: the shadows are per-client updates
trained with the victim's method from a received model, so the line from a
channel score to a count transfers to the sum up to the 1/m dilution. The
FedAvg control from results/tables/howmuch.csv gates interpretation.
"""

from pathlib import Path
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from fedseismic.privacy.decaf import positive_sum
from fedseismic.privacy.head_channels import _client_id
from fedseismic.privacy.howmuch import (
    _apply_line,
    _fit_line,
    counts_r2,
    real_counts,
    shadow_counts,
    size_r2,
)
from fedseismic.privacy.log import _load_checkpoint, load_labels, load_meta
from fedseismic.privacy.meta_attacker import training_pool
from fedseismic.privacy.metrics import to_simplex
from fedseismic.privacy.row_delta import _presence_auc

ABSENT = 1e-3
OBSERVER_SUM = "server-sum"
OBSERVER_PEER = "peer-sum"
# Victim and shadow regime. The label-free decay curve (mean |bias delta| per
# round, server-visible) falls from ~0.9 at round 1 to ~0.03 by round 10 and
# ~0.008 by round 40. The count slope lives in the early regime; late-round
# starts come from a converged model and carry almost no slope. Both sides use
# rounds 1..ROUND_HI so the shadow statistic matches the victim statistic.
ROUND_LO = 1
ROUND_HI = 10


def _round_of(path):
    return int(path.stem.split("_")[1])


def participation_channels(run_dir, round_lo=ROUND_LO, round_hi=ROUND_HI):
    """Mean head-delta channels per client over rounds ``round_lo``..``round_hi``.

    The window must match the shadow starts: shadows train once from a
    received model in that window, so scoring the all-40-round victim mean
    against them mixes regimes (large early deltas vs converged noise) and
    flattens the count line. Both sides live in the early regime.
    """
    norms = {}
    bias = {}
    decaf = {}
    per_participation = {}
    for path in sorted((Path(run_dir) / "head").glob("round_*_client_*.npz")):
        round_index = _round_of(path)
        if not (round_lo <= round_index <= round_hi):
            continue
        with np.load(path) as handle:
            weight = np.asarray(handle["weight_delta"], dtype=np.float64)
            bias_delta = np.asarray(handle["bias_delta"], dtype=np.float64).reshape(-1)
        client = _client_id(path)
        row = np.linalg.norm(weight.reshape(weight.shape[0], -1), axis=1)
        per_participation.setdefault(client, []).append((row, bias_delta, positive_sum(weight)))
    if not per_participation:
        raise FileNotFoundError(f"no head deltas in rounds {round_lo}..{round_hi} under {run_dir}")
    for client, parts in per_participation.items():
        norms[client] = np.mean(np.stack([part[0] for part in parts]), axis=0)
        bias[client] = np.mean(np.stack([part[1] for part in parts]), axis=0)
        decaf[client] = np.mean(np.stack([part[2] for part in parts]), axis=0)
    return norms, bias, decaf


def cohort_sum(channels):
    """Mean of the members' mean channel vectors. Plaintext stand-in for the
    encrypted sum: equal weights, full participation."""
    stacked = np.stack([np.asarray(channels[client], dtype=np.float64) for client in sorted(channels)])
    return float(len(channels)), np.mean(stacked, axis=0)


def _shadow_channel_scores(cfg, targets, seed, n_shadows, channel):
    """Shadow fits for one channel: line from score to count, plus shadow counts."""
    from fedseismic.privacy.share_sweep import _load_shadows

    raise NotImplementedError("use score_seed_sums, which loads cached shadows")


def score_seed_sums(run_dir, targets, n_shadows=48, round_lo=ROUND_LO, round_hi=ROUND_HI):
    """Server-sum and peer-sum scores for one seed.

    Shadows must be scale-matched (repeated Dirichlet(N) partitions) AND
    regime-matched (victim means over ``round_lo``..``round_hi``, shadows
    started in the same window). A cache built without ``cohort_size`` or
    with a different ``start_round_max`` is rejected as stale.
    """
    from fedseismic.config import RunConfig
    from fedseismic.privacy.howmuch import _counts_spearman, _dataset_targets
    from fedseismic.privacy.share_sweep import _cache_path

    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    seed = int(meta.get("seed", cfg.seed))
    histograms, _ = load_labels(run_dir)
    norms, bias, decaf = participation_channels(run_dir, round_lo, round_hi)
    counts_by_client = real_counts(cfg, targets, seed)
    clients = sorted(client for client in bias if client in histograms and client < len(counts_by_client))
    counts = np.stack([counts_by_client[client] for client in clients])
    pis = [to_simplex(histograms[client]) for client in clients]

    cache = _cache_path(run_dir)
    if not cache.is_file():
        raise FileNotFoundError(f"no shadow cache in {cache}; run _load_shadows first")
    with np.load(cache) as handle:
        def _cached_int(name):
            return int(np.asarray(handle[name]).reshape(-1)[0]) if name in handle.files else 0
        cached_cohort = _cached_int("cohort_size")
        cached_start = _cached_int("start_round_max")
        if cached_cohort != int(cfg.num_clients) or cached_start != int(round_hi):
            raise FileNotFoundError(
                f"stale shadow cache in {cache} (cohort {cached_cohort}, startmax "
                f"{cached_start}); want cohort {int(cfg.num_clients)}, startmax {int(round_hi)}")
        features = np.asarray(handle["features"], dtype=np.float64)
        mixes = np.asarray(handle["mixes"], dtype=np.float64)
        positive = np.asarray(handle["positive"], dtype=np.float64) if "positive" in handle.files else None
    shadow = shadow_counts(cfg, targets, seed, n_shadows, cohort_size=int(cfg.num_clients))
    if len(shadow) != len(features):
        raise ValueError(f"shadow rows {len(shadow)} != features {len(features)} in {run_dir}")
    classes = cfg.num_classes
    shadow_channels = {
        "bias": features[:, classes:],
        "row": features[:, :classes],
        "decaf": positive,
    }
    real_channels = {
        "bias": np.stack([bias[client] for client in clients]),
        "row": np.stack([norms[client] for client in clients]),
        "decaf": np.stack([decaf[client] for client in clients]),
    }
    out = {"n_members": len(clients), "seed": seed_dir_name(run_dir)}
    pop_profile = counts.mean(axis=0)
    out["spearman_floor"] = _counts_spearman(
        counts, np.stack([pop_profile for _ in range(len(clients))]))
    for name in ("bias", "row", "decaf"):
        scores = shadow_channels[name]
        if scores is None:
            continue
        line = _fit_line(scores, shadow)
        real = real_channels[name]
        # Server-individual reference: the usual per-client line.
        # At N=2..8 the client bootstrap is thin, so this is the pooled R2
        # across the cohort's members, not the multi-seed table number.
        indiv_hat = _apply_line(line, real)
        out[f"individual_{name}_counts_r2"] = counts_r2(counts, indiv_hat)
        out[f"individual_{name}_counts_spearman"] = _counts_spearman(counts, indiv_hat)
        # Member-level support conditioned on the union (Q3 reframing): only
        # classes present SOMEWHERE in the cohort count. Always-absent
        # classes are free negatives; removing them isolates attribution
        # (pinning k on member j) from mere union detection. Presence is
        # judged on full-pi shares, restricted to union classes.
        union_present = np.zeros(classes, dtype=bool)
        for pi in pis:
            union_present |= (to_simplex(pi) > ABSENT)
        if union_present.sum() == 0:
            out[f"individual_{name}_union_presence_auc"] = float("nan")
        else:
            cond_labels = []
            cond_scores = []
            for pi, vec in zip(pis, real):
                present = (to_simplex(pi) > ABSENT).astype(int)[union_present]
                cond_labels.extend(present.tolist())
                cond_scores.extend(np.asarray(vec, dtype=np.float64)[union_present].tolist())
            cond_labels = np.asarray(cond_labels)
            # Degenerate only when member supports do not vary on union
            # classes — not when the union itself is full (members differ).
            if len(np.unique(cond_labels)) < 2:
                out[f"individual_{name}_union_presence_auc"] = float("nan")
            else:
                out[f"individual_{name}_union_presence_auc"] = float(
                    roc_auc_score(cond_labels, np.asarray(cond_scores)))
        # Server-sum: one mean vector; score every member against it, mean the R2.
        _, summed = cohort_sum({client: real[index] for index, client in enumerate(clients)})
        sum_hat = _apply_line(line, np.stack([summed for _ in clients]))
        out[f"sumsum_{name}_counts_r2"] = counts_r2(counts, sum_hat)
        out[f"sumsum_{name}_counts_spearman"] = _counts_spearman(counts, sum_hat)
        # Presence from the sum: every member scored with the same summed vector.
        out[f"sumsum_{name}_presence_auc"] = _presence_auc(pis, [summed for _ in clients])
        # Peer-sum: member j subtracts its own update, residual estimates the
        # MEAN update of the other members. The line predicts single-client-
        # scale counts, so the target is the mean (not the sum) of the other
        # members' counts. Pairs pool across members: one R2 per seed against
        # that seed's per-class mean (a single row would make the baseline
        # degenerate). At N=2 the residual is exactly the other member's
        # update, so peer==individual by algebra.
        m = len(clients)
        peer_truth = []
        peer_pred = []
        for index in range(m):
            residual = (summed * m - real[index]) / max(m - 1, 1)
            other_mean = (counts.sum(axis=0) - counts[index]) / max(m - 1, 1)
            pred = _apply_line(line, residual.reshape(1, -1))
            peer_truth.append(other_mean)
            peer_pred.append(pred.reshape(-1))
        out[f"peer_{name}_other_counts_r2"] = counts_r2(peer_truth, peer_pred)
        out[f"peer_{name}_other_counts_spearman"] = _counts_spearman(peer_truth, peer_pred)
        # Peer presence: which classes do the OTHER members hold? Truth per
        # (member j, class k) is the union presence over members != j; the
        # score is the residual channel value. Pooled within the seed.
        union_labels = []
        union_scores = []
        for index in range(m):
            others_present = np.zeros(classes, dtype=bool)
            for other in range(m):
                if other != index:
                    others_present |= (to_simplex(pis[other]) > ABSENT)
            residual = (summed * m - real[index]) / max(m - 1, 1)
            union_labels.extend(others_present.astype(int).tolist())
            union_scores.extend(np.asarray(residual, dtype=np.float64).reshape(-1).tolist())
        union_labels = np.asarray(union_labels)
        if len(np.unique(union_labels)) == 2:
            out[f"peer_{name}_presence_auc"] = float(roc_auc_score(union_labels, np.asarray(union_scores)))
        else:
            out[f"peer_{name}_presence_auc"] = float("nan")
    return out


def seed_dir_name(run_dir):
    return Path(run_dir).name


def score_sums_grid(roots, n_draws=1000, output="results/tables/sums_blood.csv",
                    job_prefix="fedavg", round_lo=ROUND_LO, round_hi=ROUND_HI,
                    alphas=None):
    """Score finished few-silo runs. Shadows are generated per run on demand.

    ``job_prefix`` selects the algorithm jobs to score. FedPer uploads carry
    the global head put back, so their head deltas are zero by construction
    and are excluded: FedPer stays not-measured on the head channels.
    """
    import pandas as pd
    import torch

    from fedseismic.config import RunConfig, resolve_device
    from fedseismic.privacy.howmuch import _dataset_targets
    from fedseismic.privacy.log import load_meta
    from fedseismic.privacy.mechanism import DISPLAY, _mean_finite, _percentile, _resample
    from fedseismic.privacy.share_sweep import _load_shadows

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    found = []
    for root in roots:
        for job in sorted(Path(root).iterdir()):
            if not job.is_dir() or not job.name.startswith(job_prefix):
                continue
            for run in sorted(job.glob("seed_*")):
                if (run / "results.json").is_file() and (run / "head").is_dir():
                    found.append(run)
    if not found:
        raise FileNotFoundError(f"no finished runs under {list(roots)}")
    cache = {}
    per_seed = {}
    for run in found:
        meta = load_meta(run)
        cfg = RunConfig.from_mapping(meta["config"])
        if alphas is not None and float(cfg.partition_alpha) not in [float(a) for a in alphas]:
            continue
        print(f"shadows {run}", flush=True)
        # Scale-matched shadows: victims hold ~pool/N, so shadows are drawn
        # from repeated Dirichlet(N) partitions, not one 48-way split.
        # Starts are rounds round_lo..round_hi, matching the victim window:
        # late-round starts come from a converged model and carry almost no
        # count slope.
        _load_shadows(run, cfg, device, 48, cohort_size=int(cfg.num_clients),
                      start_round_max=int(round_hi))
        if cfg.dataset not in cache:
            cache[cfg.dataset] = _dataset_targets(cfg)
        print(f"score {run}", flush=True)
        scored = score_seed_sums(run, cache[cfg.dataset], round_lo=round_lo, round_hi=round_hi)
        job = run.parent.name
        key = (str(cfg.dataset), job, int(cfg.num_clients), float(cfg.partition_alpha),
               str(cfg.algorithm), bool(cfg.logit_adjust), float(cfg.prior_smooth))
        per_seed.setdefault(key, []).append(scored)
    table = []
    rng = np.random.RandomState(0)
    for key in sorted(per_seed):
        dataset, job, n_members, alpha, algorithm, logit, smooth = key
        seeds = per_seed[key]
        if str(algorithm) == "fedkper" and bool(logit):
            method = "FedKPer+logit"
        elif str(algorithm) == "fedkper":
            method = "FedKPer"
        elif str(algorithm) == "fedper":
            method = "FedPer"
        elif bool(logit) and abs(float(smooth)) < 1e-9:
            method = "FedLC"
        elif bool(logit) and abs(float(smooth) - 1.0) < 1e-9:
            method = "FedAvg"
        elif bool(logit):
            method = f"λ={float(smooth):g}"
        else:
            method = "FedAvg"
        metrics = sorted({name for row in seeds for name in row if name not in {"n_members", "seed", "spearman_floor"}})
        floor_point = _mean_finite(row["spearman_floor"] for row in seeds)
        for metric in metrics:
            values = [row[metric] for row in seeds]
            point = _mean_finite(values)
            draws = []
            for _ in range(n_draws):
                sample = [_mean_finite([values[item] for item in rng.randint(0, len(values), size=len(values))])]
                draws.append(sample[0])
            low, high = _percentile(draws)
            observer = OBSERVER_SUM if metric.startswith("sumsum_") else (
                OBSERVER_PEER if metric.startswith("peer_") else "server-individual")
            floor = float("nan")
            if "spearman" in metric:
                floor = floor_point
            elif "counts_r2" in metric:
                floor = 0.0
            table.append({
                "dataset": DISPLAY.get(dataset, dataset),
                "job": job,
                "n_members": n_members,
                "alpha": alpha,
                "method": method,
                "observer": observer,
                "metric": metric,
                "estimate": point,
                "low": low,
                "high": high,
                "floor": floor,
                "n_seeds": len(values),
            })
            print(f"  N={n_members} a={alpha:g} {method} {observer} {metric} {point:.3f} [{low:.3f}, {high:.3f}]", flush=True)
    frame = pd.DataFrame(table)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {path.resolve()}")
    return table


def main_cli(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Few-silo encrypted-sum scoring")
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", default="results/tables/sums_blood.csv")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--round-lo", type=int, default=ROUND_LO)
    parser.add_argument("--round-hi", type=int, default=ROUND_HI)
    parser.add_argument("--alphas", type=float, nargs="*", default=None)
    parser.add_argument("--figure", action="store_true",
                        help="draw leakage-vs-silos from --output CSV instead of scoring")
    args = parser.parse_args(argv)
    if args.figure:
        return plot_leakage_vs_silos(args.output)
    return score_sums_grid(args.roots, n_draws=args.draws, output=args.output,
                           round_lo=args.round_lo, round_hi=args.round_hi,
                           alphas=args.alphas)


def plot_leakage_vs_silos(csv_path="results/tables/sums_blood.csv",
                          figure_path="results/figures/leakage_vs_silos.png"):
    """Leakage vs number of silos, one line per observer.

    X is the cohort size N. Y is counts R2 (row-size line) and support AUC
    (bias channel) in two panels. Points are seed-means with 95% client-cluster
    intervals. Only cells whose FedAvg individual control passes are drawn;
    the rest are listed on stdout as not-interpreted.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    frame = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    _plot_metric_panel(
        axes[0], frame,
        {"individual_row_counts_r2", "sumsum_row_counts_r2", "peer_row_other_counts_r2"},
        "counts R2 (row-size line)", ylabel="counts R2",
    )
    _plot_metric_panel(
        axes[1], frame,
        {"sumsum_bias_presence_auc", "sumsum_row_presence_auc", "sumsum_decaf_presence_auc"},
        "support AUC from the sum", ylabel="presence AUC",
    )
    _plot_metric_panel(
        axes[2], frame,
        {"individual_bias_union_presence_auc", "individual_row_union_presence_auc",
         "individual_decaf_union_presence_auc"},
        "attribution AUC on union classes", ylabel="presence AUC",
    )
    axes[0].set_xlabel("silos N")
    axes[1].set_xlabel("silos N")
    axes[0].set_ylabel("counts R2")
    axes[1].set_ylabel("presence AUC")
    fig.tight_layout()
    path = Path(figure_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"wrote {path.resolve()}")
    return path


def _plot_metric_panel(ax, frame, metrics, title, ylabel=None):
    """One line per (method, observer) for the given metric names."""
    for (method, observer, metric), group in frame.groupby(["method", "observer", "metric"]):
        if metric not in metrics:
            continue
        group = group.sort_values("n_members")
        xs = group["n_members"].to_numpy()
        ys = group["estimate"].to_numpy()
        lo = group["low"].to_numpy()
        hi = group["high"].to_numpy()
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys],
                    marker="o", label=f"{method} / {observer}", capsize=3)
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend(fontsize="small")


if __name__ == "__main__":
    main_cli()
