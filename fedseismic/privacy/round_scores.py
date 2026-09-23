"""Advantage of each upload over the global model that client actually received."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from fedseismic.privacy.log import load_labels
from fedseismic.privacy.metrics import leakage_from_tv, to_simplex, total_variation


def _tpr_at_fpr(labels, scores, target=0.01):
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target, fpr, tpr))


def score_run(run_dir):
    run_dir = Path(run_dir)
    path = run_dir / "round_scores.jsonl"
    histograms, _ = load_labels(run_dir)
    by_client = {}
    presence = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            client = int(record["client"])
            pi = to_simplex(histograms[client])
            local_hat = np.asarray(record["local_softmax"], dtype=np.float64)
            global_hat = np.asarray(record["global_softmax"], dtype=np.float64)
            advantage = (
                leakage_from_tv(total_variation(pi, local_hat))
                - leakage_from_tv(total_variation(pi, global_hat))
            )
            by_client.setdefault(client, []).append((int(record["round"]), advantage))
            for class_index, mass in enumerate(pi):
                presence.append({
                    "client": client,
                    "round": int(record["round"]),
                    "class_index": class_index,
                    "present": int(mass > 1e-3),
                    "residual": float(local_hat[class_index] - global_hat[class_index]),
                })
    last = []
    averaged = []
    for advantages in by_client.values():
        advantages.sort()
        last.append(advantages[-1][1])
        averaged.append(float(np.mean([value for _, value in advantages])))
    return {
        "last_advantage": float(np.mean(last)) if last else float("nan"),
        "mean_advantage": float(np.mean(averaged)) if averaged else float("nan"),
        "presence": presence,
    }


def load_advantages(run_dir):
    """One row per participation. Advantage is upload leakage minus the received global."""
    run_dir = Path(run_dir)
    histograms, _ = load_labels(run_dir)
    rows = []
    with (run_dir / "round_scores.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            client = int(record["client"])
            pi = to_simplex(histograms[client])
            local_hat = np.asarray(record["local_softmax"], dtype=np.float64)
            global_hat = np.asarray(record["global_softmax"], dtype=np.float64)
            rows.append({
                "round": int(record["round"]),
                "client": client,
                "advantage": (
                    leakage_from_tv(total_variation(pi, local_hat))
                    - leakage_from_tv(total_variation(pi, global_hat))
                ),
            })
    return pd.DataFrame(rows)


def late_last(frame, window=10):
    """Last participation of each client inside the final ``window`` rounds."""
    if frame.empty:
        return frame
    late = frame[frame["round"] >= int(frame["round"].max()) - (window - 1)]
    return late.sort_values("round").groupby("client", as_index=False).tail(1)


def residual_dprime(run_dir, threshold=1e-3):
    """Per-class d′ of the softmax residual, present minus absent.

    Rows are participations, so they are not independent samples.
    """
    run_dir = Path(run_dir)
    histograms, _ = load_labels(run_dir)
    grouped = {}
    with (run_dir / "round_scores.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            pi = to_simplex(histograms[int(record["client"])])
            local_hat = np.asarray(record["local_softmax"], dtype=np.float64)
            global_hat = np.asarray(record["global_softmax"], dtype=np.float64)
            residual = local_hat - global_hat
            for class_index, mass in enumerate(pi):
                grouped.setdefault(class_index, {"pos": [], "neg": []})
                bucket = "pos" if mass > threshold else "neg"
                grouped[class_index][bucket].append(float(residual[class_index]))
    rows = []
    for class_index, buckets in sorted(grouped.items()):
        pos = np.asarray(buckets["pos"], dtype=np.float64)
        neg = np.asarray(buckets["neg"], dtype=np.float64)
        if pos.size < 2 or neg.size < 2:
            value = float("nan")
        else:
            pooled = np.sqrt(
                ((pos.size - 1) * pos.var(ddof=1) + (neg.size - 1) * neg.var(ddof=1))
                / (pos.size + neg.size - 2)
            )
            value = float((pos.mean() - neg.mean()) / pooled) if pooled > 0 else float("nan")
        rows.append({
            "class_index": class_index,
            "d_prime": value,
            "n_pos": int(pos.size),
            "n_neg": int(neg.size),
        })
    return rows


def client_bootstrap(values, draws=2000, rng=None):
    """Percentile interval of the client-mean, resampling clients."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(0) if rng is None else rng
    picked = rng.integers(0, values.size, size=(draws, values.size))
    means = values[picked].mean(axis=1)
    low, mid, high = np.percentile(means, [2.5, 50, 97.5])
    return float(low), float(mid), float(high)


def summarize_root(root, dataset):
    root = Path(root)
    rows = []
    presence_rows = []
    for scores in sorted(root.glob("*/seed_*/round_scores.jsonl")):
        run_dir = scores.parent
        meta_name = run_dir.parent.name
        result = score_run(run_dir)
        rows.append({
            "dataset": dataset,
            "job": meta_name,
            "seed": run_dir.name,
            "last_advantage": result["last_advantage"],
            "mean_advantage": result["mean_advantage"],
        })
        for item in result["presence"]:
            item["dataset"] = dataset
            item["job"] = meta_name
            item["seed"] = run_dir.name
            presence_rows.append(item)
    frame = pd.DataFrame(rows)
    presence = pd.DataFrame(presence_rows)
    tpr_rows = []
    if not presence.empty:
        for (job, class_index), part in presence.groupby(["job", "class_index"]):
            labels = part["present"].to_numpy()
            scores = part["residual"].to_numpy()
            tpr_rows.append({
                "dataset": dataset,
                "job": job,
                "class_index": int(class_index),
                "n": int(len(part)),
                "n_neg": int((labels == 0).sum()),
                "auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan"),
                "tpr_at_1": _tpr_at_fpr(labels, scores),
            })
    return frame, pd.DataFrame(tpr_rows)
