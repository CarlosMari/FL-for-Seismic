"""Distribution inference from an upload the server already stores.

The vector attack reads the class mix off the public softmax. The scalar
attack uses only the normalized entropy of that softmax, which is the mix
summary FedKPer multiplies into the aggregation weight. A one-number summary
can track how skewed a client is. It cannot name which class is present when
every client is skewed by a different class.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score, roc_curve

from fedseismic.federated.aggregation import normalized_entropy
from fedseismic.privacy.log import load_labels
from fedseismic.privacy.metrics import leakage_from_tv, to_simplex, total_variation


def _tpr_at_fpr(labels, scores, target=0.01):
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target, fpr, tpr))


def histogram_r2(true, predicted):
    """R² of predicted class fractions against the true mix, classes pooled."""
    left = np.concatenate([to_simplex(row) for row in true])
    right = np.concatenate([to_simplex(row) for row in predicted])
    return float(r2_score(left, right))


def entropy_r2(true, predicted):
    """R² of normalized entropy. This is skewness, not which class."""
    y_true = np.asarray([normalized_entropy(row) for row in true], dtype=np.float64)
    y_hat = np.asarray([normalized_entropy(row) for row in predicted], dtype=np.float64)
    if np.std(y_true) < 1e-12:
        return float("nan")
    return float(r2_score(y_true, y_hat))


def _mean_histogram(rows):
    stacked = np.stack([to_simplex(row) for row in rows])
    return to_simplex(stacked.mean(axis=0))


def scalar_histogram_attack(train_entropy, train_pi, test_entropy):
    """Map one entropy number onto a histogram with a ridge fit on other clients.

    The fit sees those clients' true mixes, so this is an optimistic server.
    A constant entropy across classes cannot recover a rotated one-hot.
    """
    features = np.asarray(train_entropy, dtype=np.float64).reshape(-1, 1)
    target = np.stack([to_simplex(row) for row in train_pi])
    model = Ridge(alpha=1.0)
    model.fit(features, target)
    predicted = model.predict(np.asarray(test_entropy, dtype=np.float64).reshape(-1, 1))
    return [to_simplex(np.clip(row, 0.0, None)) for row in predicted]


def leave_client_advantage(by_client):
    """Last-upload advantage of the entropy ridge and of the softmax itself.

    ``by_client`` maps a client id to a list of ``(round, pi, local_hat, global_hat)``.
    The held-out client's last round is scored. The ridge is fit on every other
    client's participations.
    """
    clients = sorted(by_client)
    if len(clients) < 3:
        return None
    softmax_gaps = []
    scalar_gaps = []
    softmax_hats = []
    scalar_hats = []
    truths = []
    for held in clients:
        series = sorted(by_client[held], key=lambda item: item[0])
        _, pi, local_hat, global_hat = series[-1]
        train = [
            item for client, series_ in by_client.items() if client != held for item in series_
        ]
        predicted = scalar_histogram_attack(
            [normalized_entropy(item[2]) for item in train],
            [item[1] for item in train],
            [normalized_entropy(local_hat)],
        )[0]
        baseline = _mean_histogram([item[1] for item in train])
        truths.append(pi)
        softmax_hats.append(local_hat)
        scalar_hats.append(predicted)
        softmax_gaps.append(
            leakage_from_tv(total_variation(pi, local_hat))
            - leakage_from_tv(total_variation(pi, global_hat))
        )
        scalar_gaps.append(
            leakage_from_tv(total_variation(pi, predicted))
            - leakage_from_tv(total_variation(pi, baseline))
        )
    return {
        "softmax_advantage": float(np.mean(softmax_gaps)),
        "scalar_advantage": float(np.mean(scalar_gaps)),
        "softmax_r2": histogram_r2(truths, softmax_hats),
        "scalar_r2": histogram_r2(truths, scalar_hats),
        "entropy_r2": entropy_r2(truths, softmax_hats),
        "n_clients": len(clients),
    }


def presence_table(frame, min_negatives=50):
    """Per-class presence AUC from the upload residual, one row per class."""
    rows = []
    for class_index, part in frame.groupby("class_index"):
        labels = part["present"].to_numpy()
        scores = part["residual"].to_numpy()
        n_neg = int((labels == 0).sum())
        n_pos = int((labels == 1).sum())
        if len(np.unique(labels)) < 2:
            auc = float("nan")
            tpr = float("nan")
        else:
            auc = float(roc_auc_score(labels, scores))
            tpr = _tpr_at_fpr(labels, scores)
        rows.append({
            "class_index": int(class_index),
            "presence_auc": auc,
            "tpr_at_1": tpr,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "reliable": bool(n_neg >= min_negatives and n_pos >= min_negatives),
        })
    return pd.DataFrame(rows)


def load_round_records(run_dir):
    """One participation: true mix, upload softmax, and the received global."""
    run_dir = Path(run_dir)
    histograms, _ = load_labels(run_dir)
    by_client = {}
    rows = []
    with (run_dir / "round_scores.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            client = int(record["client"])
            pi = to_simplex(histograms[client])
            local_hat = np.asarray(record["local_softmax"], dtype=np.float64)
            global_hat = np.asarray(record["global_softmax"], dtype=np.float64)
            by_client.setdefault(client, []).append(
                (int(record["round"]), pi, local_hat, global_hat)
            )
            for class_index, mass in enumerate(pi):
                rows.append({
                    "client": client,
                    "round": int(record["round"]),
                    "class_index": int(class_index),
                    "present": int(mass > 1e-3),
                    "residual": float(local_hat[class_index] - global_hat[class_index]),
                })
    return by_client, pd.DataFrame(rows)
