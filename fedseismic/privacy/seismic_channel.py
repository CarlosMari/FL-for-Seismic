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

    One observation per (round, site) from ``view/``. Cosine distance
    between running-mean stacks; same-site pairs are positives. High AUC
    means the uploaded statistics identify the site. Needs no labels.
    """
    vectors = {}
    for path in sorted((Path(run_dir) / "view").glob("round_*_client_*.npz")):
        with np.load(path) as handle:
            if "bn_keys" not in handle.files:
                return float("nan"), 0
            parts = []
            for key in sorted(str(k) for k in handle["bn_keys"]):
                if "running_mean" in key:
                    parts.append(np.asarray(handle[f"bn::{key}"], dtype=np.float64).reshape(-1))
        if not parts:
            return float("nan"), 0
        round_index = int(path.stem.split("_")[1])
        client = int(path.stem.split("_")[-1])
        vectors.setdefault(client, []).append(np.concatenate(parts))
    clients = sorted(vectors)
    labels, scores = [], []
    items = [(client, vec) for client in clients for vec in vectors[client]]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            same = 1 if items[i][0] == items[j][0] else 0
            left, right = items[i][1], items[j][1]
            denom = float(np.linalg.norm(left) * np.linalg.norm(right))
            cosine = float(np.dot(left, right) / denom) if denom > 1e-18 else 0.0
            labels.append(same)
            scores.append(cosine)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan"), len(labels)
    return float(roc_auc_score(labels, np.asarray(scores))), len(labels)


def score_job(job_dir):
    """Hand-built channels per seed, pooled over sites."""
    counts_all, rows_all, bias_all, pis_all = [], [], [], []
    bn_aucs = []
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
        auc, _ = bn_site_auc(run)
        bn_aucs.append(auc)
    counts_all = np.concatenate(counts_all)
    rows_all = np.concatenate(rows_all)
    bias_all = np.concatenate(bias_all)
    pis_all = np.concatenate(pis_all)
    # Pearson on log-counts: pixel counts span orders of magnitude and the
    # mechanism is multiplicative in n_k; raw-count Pearson would be
    # dominated by the biggest class.
    log_counts = np.log10(np.maximum(counts_all, 1.0))
    return {
        "row_logcount_pearson": _pearson(log_counts, rows_all),
        "bias_logcount_pearson": _pearson(log_counts, bias_all),
        "row_presence_auc": _presence_auc(pis_all, rows_all),
        "bias_presence_auc": _presence_auc(pis_all, bias_all),
        "bn_site_auc": float(np.nanmean(bn_aucs)) if bn_aucs else float("nan"),
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
        utility = json.loads((sorted(job.glob("seed_*"))[0] / "results.json").read_text())
        locals_mean, globals_mean = [], []
        for run in sorted(job.glob("seed_*")):
            um = json.loads((run / "results.json").read_text())
            locals_mean.append(um["local_mean"])
            globals_mean.append(um["final"])
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
