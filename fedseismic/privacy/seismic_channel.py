"""First seismic read: site facies counts, outc-head channels, BN statistics.

Uses the recovered last-round outc deltas (``head_outc/``, see
``seismic_head_fix``): per-round outc deltas for rounds 1-19 are
unrecoverable, so this is one round per site, not a 40-round mean. All
scores below are hand-built (Pearson, presence AUC) — no shadow fits, so
no counts R² yet. UNet shadows (5 epochs × 48 per run) are queued only if
the hand-built control passes.

``how much`` here is per-site facies PIXEL counts on the train split.
"""

from pathlib import Path
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from fedseismic.privacy.log import load_labels, load_meta
from fedseismic.privacy.metrics import to_simplex

ABSENT = 1e-3
N_CLASSES = 6


def site_train_lines(num_crosslines=701, num_clients=3, local_test_ratio=0.2):
    """Train crossline counts per site, mirroring the contiguous split."""
    chunk = num_crosslines // num_clients
    sizes = []
    for client in range(num_clients):
        start = client * chunk
        end = (client + 1) * chunk if client < num_clients - 1 else num_crosslines
        length = end - start
        n_test = max(1, int(round(length * local_test_ratio)))
        n_test = min(n_test, length - 1)
        sizes.append(length - n_test)
    return sizes


def site_counts(run_dir):
    """Pixel counts per (site, facies) on the train split."""
    from fedseismic.config import RunConfig

    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    histograms, _ = load_labels(run_dir)
    lines = site_train_lines(701, cfg.num_clients, cfg.local_test_ratio)
    pixels_per_line = 401 * 255
    counts = []
    for client in sorted(histograms):
        fracs = to_simplex(histograms[client])
        counts.append(fracs * lines[client] * pixels_per_line)
    return np.stack(counts)


def outc_channels(run_dir, dirname="head_outc"):
    """Row sizes and bias of the recovered last-round outc deltas."""
    rows, biases = {}, {}
    for path in sorted((Path(run_dir) / dirname).glob("round_*_client_*.npz")):
        with np.load(path) as handle:
            weight = np.asarray(handle["weight_delta"], dtype=np.float64)
            bias = np.asarray(handle["bias_delta"], dtype=np.float64).reshape(-1)
        client = int(path.stem.split("_")[-1])
        rows[client] = np.linalg.norm(weight.reshape(weight.shape[0], -1), axis=1)
        biases[client] = bias
    return rows, biases


def _pearson(truths, scores):
    from scipy.stats import pearsonr

    left = np.concatenate([np.asarray(t, dtype=np.float64).reshape(-1) for t in truths])
    right = np.concatenate([np.asarray(s, dtype=np.float64).reshape(-1) for s in scores])
    if np.std(right) < 1e-12:
        return float("nan")
    coef = pearsonr(left, right).statistic
    return float(coef) if np.isfinite(coef) else float("nan")


def _presence_auc(truths, scores, threshold=ABSENT):
    labels = []
    values = []
    for truth, score in zip(truths, scores):
        present = (np.asarray(truth, dtype=np.float64).reshape(-1) > threshold).astype(int)
        labels.extend(present.tolist())
        values.extend(np.asarray(score, dtype=np.float64).reshape(-1).tolist())
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(values)))


def bn_site_auc(run_dir, last_round=20):
    """Same-site vs different-site AUC from BN running-mean vectors.

    Centered per (round, layer) across sites, z-scored per channel, then
    pooled over all (round, site) pairs. Centering is essential: all sites
    share a large common mean that saturates raw cosine near 1 (the naive
    pooled metric sits at 0.56). Needs no labels. Returns overall, early
    (rounds 1-10), late (11+), and first-encoder-layer AUCs plus pair count.
    """
    obs = {}
    for path in sorted((Path(run_dir) / "view").glob("round_*_client_*.npz")):
        with np.load(path) as handle:
            if "bn_keys" not in handle.files:
                continue
            round_index = int(path.stem.split("_")[1])
            client = int(path.stem.split("_")[-1])
            layers = {}
            for key in handle["bn_keys"]:
                key = str(key)
                if "running_mean" in key:
                    layers[key.split(".running_mean")[0]] = np.asarray(
                        handle[f"bn::{key}"], dtype=np.float64).reshape(-1)
            if layers:
                obs.setdefault(round_index, {})[client] = layers
    if not obs:
        return {"all": float("nan"), "early": float("nan"), "late": float("nan"),
                "first": float("nan"), "pairs": 0}
    layers = sorted(next(iter(next(iter(obs.values())).values())))
    centered = {}
    for round_index, clients in obs.items():
        ordered = sorted(clients)
        for layer in layers:
            matrix = np.stack([clients[client][layer] for client in ordered])
            mean, std = matrix.mean(axis=0), matrix.std(axis=0) + 1e-12
            for pos, client in enumerate(ordered):
                centered.setdefault(layer, {})[(round_index, client)] = (matrix[pos] - mean) / std

    def auc_for(select):
        labels, scores = [], []
        for layer in layers:
            items = [(key, vec) for key, vec in centered[layer].items() if select(key[0])]
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    same = 1 if items[i][0][1] == items[j][0][1] else 0
                    left, right = items[i][1], items[j][1]
                    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
                    cosine = float(np.dot(left, right) / denom) if denom > 1e-18 else 0.0
                    labels.append(same)
                    scores.append(cosine)
        labels = np.asarray(labels)
        if len(np.unique(labels)) < 2:
            return float("nan"), 0
        return float(roc_auc_score(labels, np.asarray(scores))), len(labels)

    all_auc, pairs = auc_for(lambda r: True)
    early_auc, _ = auc_for(lambda r: r <= 10)
    late_auc, _ = auc_for(lambda r: r > 10)
    first_items = sorted(centered[layers[0]].items())
    first_labels, first_scores = [], []
    for i in range(len(first_items)):
        for j in range(i + 1, len(first_items)):
            first_labels.append(1 if first_items[i][0][1] == first_items[j][0][1] else 0)
            left, right = first_items[i][1], first_items[j][1]
            denom = float(np.linalg.norm(left) * np.linalg.norm(right))
            first_scores.append(float(np.dot(left, right) / denom) if denom > 1e-18 else 0.0)
    first_auc = float(roc_auc_score(first_labels, np.asarray(first_scores)))
    return {"all": all_auc, "early": early_auc, "late": late_auc,
            "first": first_auc, "pairs": pairs}


def _within_pearson(log_counts, scores):
    """Mean per-class Pearson after centering each class across sites.

    The pooled Pearson is dominated by class identity (big classes are big
    everywhere). Centering removes that and asks whether sites with more of
    class k show larger updates for k.
    """
    from scipy.stats import pearsonr

    values = []
    for k in range(log_counts.shape[1]):
        x, y = log_counts[:, k], np.asarray(scores)[:, k]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        coef = pearsonr(x - x.mean(), y - y.mean()).statistic
        if np.isfinite(coef):
            values.append(float(coef))
    if not values:
        return float("nan")
    return float(np.mean(values))


def score_job(job_dir):
    """Hand-built channels per seed, pooled over sites."""
    counts_all, rows_all, bias_all, pis_all = [], [], [], []
    bn_all, bn_early, bn_late, bn_first, bn_pairs = [], [], [], [], []
    for run in sorted(Path(job_dir).glob("seed_*")):
        if not (run / "results.json").is_file():
            continue
        counts = site_counts(run)
        rows, biases = outc_channels(run)
        histograms, _ = load_labels(run)
        clients = sorted(set(range(len(counts))) & set(rows))
        counts_all.append(counts[clients])
        rows_all.append(np.stack([rows[c] for c in clients]))
        bias_all.append(np.stack([biases[c] for c in clients]))
        pis_all.append(np.stack([to_simplex(histograms[c]) for c in clients]))
        scored = bn_site_auc(run)
        bn_all.append(scored["all"])
        bn_early.append(scored["early"])
        bn_late.append(scored["late"])
        bn_first.append(scored["first"])
        bn_pairs.append(scored["pairs"])
    counts_all = np.concatenate(counts_all)
    rows_all = np.concatenate(rows_all)
    bias_all = np.concatenate(bias_all)
    pis_all = np.concatenate(pis_all)
    # Pearson on log-counts: pixel counts span orders of magnitude and the
    # mechanism is multiplicative in n_k; raw-count Pearson would be
    # dominated by the biggest class.
    log_counts = np.log10(np.maximum(counts_all, 1.0))
    present = (pis_all > ABSENT).astype(int)
    return {
        "row_logcount_pearson": _pearson(log_counts, rows_all),
        "bias_logcount_pearson": _pearson(log_counts, bias_all),
        "row_within_pearson": _within_pearson(log_counts, rows_all),
        "bias_within_pearson": _within_pearson(log_counts, bias_all),
        "row_presence_auc": _presence_auc(pis_all, rows_all),
        "bias_presence_auc": _presence_auc(pis_all, bias_all),
        "presence_pos": int(present.sum()),
        "presence_neg": int((1 - present).sum()),
        "bn_site_auc": float(np.nanmean(bn_all)) if bn_all else float("nan"),
        "bn_site_early": float(np.nanmean(bn_early)) if bn_early else float("nan"),
        "bn_site_late": float(np.nanmean(bn_late)) if bn_late else float("nan"),
        "bn_site_first": float(np.nanmean(bn_first)) if bn_first else float("nan"),
        "bn_pairs": int(np.sum(bn_pairs)),
        "n_clients": int(counts_all.shape[0]),
    }


def main():
    import pandas as pd

    from fedseismic.config import RunConfig
    from fedseismic.privacy.log import load_meta

    root = Path("results/runs/seismic_f3")
    rows = []
    for job in sorted(root.iterdir()):
        if not job.is_dir():
            continue
        print(f"score {job.name}", flush=True)
        scored = score_job(job)
        meta = load_meta(sorted(job.glob("seed_*"))[0])
        cfg = RunConfig.from_mapping(meta["config"])
        locals_mean, globals_mean = [], []
        for run in sorted(job.glob("seed_*")):
            um = json.loads((run / "results.json").read_text())
            globals_mean.append(um["final"])
            # results.json local_mean on these runs is pixel accuracy (the
            # end-of-run scorer predates the task-metric fix); the per-round
            # history holds macro mIoU, so the local number comes from there.
            local_rounds = [
                json.loads(line)["local_mean"]
                for line in (run / "history.jsonl").read_text().splitlines()
                if "local_mean" in line
            ]
            local_rounds = [v for v in local_rounds if v is not None]
            locals_mean.append(local_rounds[-1] if local_rounds else float("nan"))
        rows.append({
            "job": job.name,
            "algorithm": cfg.algorithm,
            "norm": cfg.norm,
            "n_sites": int(cfg.num_clients),
            "local_miou": float(np.mean(locals_mean)),
            "global_miou": float(np.mean(globals_mean)),
            **scored,
        })
        print(" ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in scored.items()}, flush=True)
    frame = pd.DataFrame(rows)
    out = Path("results/tables/seismic_f3.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out.resolve()}")
    return frame


if __name__ == "__main__":
    main()
