"""Counts, dataset size, and proportions given the support.

Every number here is what a server sees in one client's update. Shadow
clients use the victim's loss and a received model from that run. The fit
never sees a real client's counts. FedAvg is scored first.
"""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from fedseismic.privacy.metrics import to_simplex
from fedseismic.privacy.row_delta import _presence_auc

ABSENT = 1e-3
OBSERVER = "server-individual"


def counts_r2(true_counts, pred_counts):
    """R² against the per-class mean count. That mean scores 0."""
    truth = np.stack([np.asarray(row, dtype=np.float64) for row in true_counts])
    pred = np.stack([np.asarray(row, dtype=np.float64) for row in pred_counts])
    if truth.shape != pred.shape:
        raise ValueError("count predictions must match the true counts")
    baseline = np.broadcast_to(truth.mean(axis=0), truth.shape)
    return _above_baseline(truth, pred, baseline)


def size_r2(true_counts, pred_sizes):
    """R² of dataset size. The mean size scores 0."""
    truth = np.asarray([float(np.asarray(row).sum()) for row in true_counts], dtype=np.float64)
    pred = np.asarray(pred_sizes, dtype=np.float64).reshape(-1)
    if len(truth) != len(pred):
        raise ValueError("size predictions must match the clients")
    if np.std(truth) < 1e-12:
        return float("nan")
    return float(r2_score(truth, pred))


def condition_on_support(pi, threshold=ABSENT):
    """π on its own support, zero on the classes it does not have."""
    share = to_simplex(pi)
    support = share > threshold
    out = np.zeros_like(share)
    if not np.any(support):
        return out
    out[support] = share[support] / share[support].sum()
    return out


def uniform_on_support(pi, threshold=ABSENT):
    """1/|S| on the true support. This is the proportions floor."""
    share = to_simplex(pi)
    support = share > threshold
    out = np.zeros_like(share)
    if not np.any(support):
        out[:] = 1.0 / len(share)
        return out
    out[support] = 1.0 / int(np.sum(support))
    return out


def renormalize(scores, mask):
    """Non-negative scores on ``mask``, made into a distribution."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    out = np.zeros(len(scores), dtype=np.float64)
    if not np.any(mask):
        out[:] = 1.0 / len(scores)
        return out
    chosen = np.clip(scores, 0.0, None)
    total = float(chosen[mask].sum())
    if total <= 1e-12:
        out[mask] = 1.0 / int(np.sum(mask))
    else:
        out[mask] = chosen[mask] / total
    return out


def proportions_r2(pis, predictions, threshold=ABSENT):
    """R² of within-support shares. Predicting 1/|S| scores 0."""
    truth = [condition_on_support(pi, threshold) for pi in pis]
    baseline = [uniform_on_support(pi, threshold) for pi in pis]
    pred = [np.asarray(row, dtype=np.float64) for row in predictions]
    return _above_baseline(np.stack(truth), np.stack(pred), np.stack(baseline))


def _above_baseline(truth, pred, baseline):
    residual = float(np.sum((truth - pred) ** 2))
    scale = float(np.sum((truth - baseline) ** 2))
    if scale < 1e-18:
        return float("nan")
    return float(1.0 - residual / scale)


def _bincount(targets, indices, num_classes):
    labels = np.asarray(targets)[list(indices)]
    return np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64)


def real_counts(cfg, targets, seed):
    """Train-split class counts, in client order, matching ``labels/pi.json``.

    Uses the same retry rule as training, so retried seeds verify.
    """
    from fedseismic.experiment import _public_pad, _train_splits_or_retry

    rng = np.random.RandomState(int(seed))
    pool, kept, _pad = _public_pad(cfg, targets, seed)
    train_parts, _test_parts = _train_splits_or_retry(cfg, pool, kept, seed, rng)
    return [_bincount(targets, part, cfg.num_classes) for part in train_parts]


def shadow_counts(cfg, targets, seed, n_shadows, cohort_size=None):
    """Class counts of the shadow clients, in the order their features were saved.

    ``cohort_size`` draws shadows from repeated Dirichlet(N) partitions so
    each shadow holds ~pool/N like the victims. It must match the
    ``cohort_size`` passed to ``collect_shadows``: both draw from the same
    RNG stream (seed 30_000 + run seed), so the counts align with the saved
    features in order. The default (None) keeps one ``n_shadows``-way split.
    """
    from fedseismic.data.cifar import partition_dirichlet
    from fedseismic.data.partition import remap_partitions
    from fedseismic.experiment import _public_pad

    pool, kept, _pad = _public_pad(cfg, targets, seed)
    rng = np.random.RandomState(30_000 + int(seed))
    if cohort_size is not None and int(cohort_size) > 0:
        size = int(cohort_size)
        draws = int(np.ceil(float(n_shadows) / size))
        parts = []
        for _ in range(draws):
            parts.extend(partition_dirichlet(pool, size, cfg.partition_alpha, rng))
    else:
        parts = partition_dirichlet(pool, n_shadows, cfg.partition_alpha, rng)
    if kept is not None:
        parts = remap_partitions(parts, kept)
    parts = [part for part in parts if len(part) >= 2][: int(n_shadows)]
    return np.stack([_bincount(targets, part, cfg.num_classes) for part in parts])


def _dataset_targets(cfg):
    name = str(cfg.dataset)
    if name == "cifar10":
        from fedseismic.data.cifar import load_cifar10
        _images, targets, _test_images, _test_targets = load_cifar10(cfg.data_root, download=False)
        return targets
    from fedseismic.data.medmnist import load_medmnist
    _images, targets, _test_images, _test_targets, _spec = load_medmnist(name, cfg.data_root, download=False)
    return targets


class _FloorLine:
    """Counts = shadow per-class mean + shrunk linear residual.

    A plain least-squares line can extrapolate far beyond the shadow range
    and score far below the floor (R² < 0) on real clients. That is a
    mis-calibrated attacker, not evidence of privacy: the attacker could
    always fall back to predicting the floor. This fit standardizes the
    single-channel scores on the shadows and fits the residual over the
    per-class mean with a cross-validated ridge penalty. With no signal the
    penalty drives the slope to zero and the prediction to the floor
    (R² = 0 against the per-class-mean baseline by construction).
    """

    def __init__(self, mean, scaler, model):
        self.mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        self.scaler = scaler
        self.model = model

    def predict(self, scores):
        flat = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        residual = np.asarray(self.model.predict(self.scaler.transform(flat))).reshape(-1)
        tiled = np.broadcast_to(self.mean, (len(residual) // len(self.mean), len(self.mean)))
        return (tiled.reshape(-1) + residual).reshape(np.asarray(scores).shape)


def _fit_line(scores, counts):
    """Floor-plus-shrunk-residual line from a channel score to a count.

    Pooled over classes: one slope maps the (standardized) score to the
    residual over the shadow per-class mean. Replaces the plain
    least-squares fit, which could extrapolate below the floor.
    """
    flat_x = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    matrix = np.asarray(counts, dtype=np.float64)
    mean = matrix.mean(axis=0)
    residual = (matrix - mean).reshape(-1)
    scaler = StandardScaler()
    model = RidgeCV(alphas=np.logspace(-6, 3, 19))
    model.fit(scaler.fit_transform(flat_x), residual)
    return _FloorLine(mean, scaler, model)


def _apply_line(model, scores):
    flat = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    return model.predict(flat).reshape(np.asarray(scores).shape)


def _fit_pair(row_scores, bias_scores, counts):
    """Pooled ridge from one class's row size and bias to that class's count.

    Features are standardized on the shadows. The penalty is chosen by
    leave-one-out on those same shadows. This is the combination of the two
    single channels: it can ignore the bias, and it does not see other classes.
    """
    features = np.column_stack([
        np.asarray(row_scores, dtype=np.float64).ravel(),
        np.asarray(bias_scores, dtype=np.float64).ravel(),
    ])
    target = np.asarray(counts, dtype=np.float64).ravel()
    scaler = StandardScaler()
    model = RidgeCV(alphas=np.logspace(-6, 3, 19))
    model.fit(scaler.fit_transform(features), target)
    return scaler, model


def _apply_pair(scaler, model, row_scores, bias_scores):
    shape = np.asarray(row_scores).shape
    features = np.column_stack([
        np.asarray(row_scores, dtype=np.float64).ravel(),
        np.asarray(bias_scores, dtype=np.float64).ravel(),
    ])
    return model.predict(scaler.transform(features)).reshape(shape)


def _fit_ridge(features, target):
    scaler = StandardScaler()
    transformed = scaler.fit_transform(np.asarray(features, dtype=np.float64))
    model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model.fit(transformed, np.asarray(target, dtype=np.float64))
    return scaler, model


def _fit_floor_ridge(features, target):
    """Mean + shrunk ridge residual, so a no-signal size fit lands on the
    mean size (R² = 0) instead of below it. Same floor-fallback argument as
    ``_FloorLine``."""
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mean = float(target.mean())
    scaler = StandardScaler()
    model = RidgeCV(alphas=np.logspace(-6, 3, 19))
    model.fit(scaler.fit_transform(np.asarray(features, dtype=np.float64)), target - mean)
    return mean, scaler, model


def _apply_floor_ridge(fit, features):
    mean, scaler, model = fit
    residual = np.asarray(model.predict(scaler.transform(np.asarray(features, dtype=np.float64)))).reshape(-1)
    return mean + residual


def _apply_ridge(scaler, model, features):
    return model.predict(scaler.transform(np.asarray(features, dtype=np.float64)))


def _fit_presence(scores, counts):
    """One logistic from a per-class score to presence, pooled over classes."""
    present = (np.asarray(counts, dtype=np.float64) > 0).astype(int).reshape(-1)
    flat = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    if len(np.unique(present)) < 2:
        return None
    model = LogisticRegression(class_weight="balanced", max_iter=500)
    model.fit(flat, present)
    return model


def _presence_from_line(model, scores):
    if model is None:
        return np.ones(np.asarray(scores).shape, dtype=bool)
    proba = model.predict_proba(np.asarray(scores, dtype=np.float64).reshape(-1, 1))[:, 1]
    return (proba > 0.5).reshape(np.asarray(scores).shape)


def _fit_vector_presence(features, counts):
    present = np.asarray(counts, dtype=np.float64) > 0
    models = []
    for class_index in range(present.shape[1]):
        labels = present[:, class_index].astype(int)
        if len(np.unique(labels)) < 2:
            models.append(None)
            continue
        model = LogisticRegression(class_weight="balanced", max_iter=500)
        model.fit(features, labels)
        models.append(model)
    return models


def _vector_support(models, features):
    features = np.asarray(features, dtype=np.float64)
    support = np.ones((len(features), len(models)), dtype=bool)
    for class_index, model in enumerate(models):
        if model is None:
            continue
        support[:, class_index] = model.predict_proba(features)[:, 1] > 0.5
    return support


def _channel_predictions(shadow_scores, shadow_counts, real_scores):
    line = _fit_line(shadow_scores, shadow_counts)
    counts_hat = _apply_line(line, real_scores)
    size_fit = _fit_floor_ridge(np.asarray(shadow_scores, dtype=np.float64), shadow_counts.sum(axis=1))
    sizes_hat = _apply_floor_ridge(size_fit, np.asarray(real_scores, dtype=np.float64))
    presence = _fit_presence(shadow_scores, shadow_counts)
    support_hat = _presence_from_line(presence, real_scores)
    return counts_hat, np.asarray(sizes_hat, dtype=np.float64).reshape(-1), support_hat


def _ridge_predictions(shadow_features, shadow_counts, real_features):
    count_scaler, count_model = _fit_ridge(shadow_features, shadow_counts)
    counts_hat = _apply_ridge(count_scaler, count_model, real_features)
    size_scaler, size_model = _fit_ridge(shadow_features, shadow_counts.sum(axis=1))
    sizes_hat = _apply_ridge(size_scaler, size_model, real_features).reshape(-1)
    support_hat = _vector_support(
        _fit_vector_presence(count_scaler.transform(shadow_features), shadow_counts),
        count_scaler.transform(real_features),
    )
    return counts_hat, sizes_hat, support_hat


def _rows_from_predictions(clients, pis, counts, counts_hat, sizes_hat, support_hat):
    rows = []
    for index, client in enumerate(clients):
        predicted = np.asarray(counts_hat[index], dtype=np.float64)
        true_support = to_simplex(pis[index]) > ABSENT
        rows.append({
            "client": int(client),
            "pi": np.asarray(pis[index], dtype=np.float64),
            "counts": np.asarray(counts[index], dtype=np.float64),
            "pred_counts": predicted,
            "pred_size": float(sizes_hat[index]),
            "oracle": renormalize(predicted, true_support),
            "estimated": renormalize(predicted, support_hat[index]),
        })
    return rows


def _counts_spearman(counts, pred):
    """Pooled Spearman rank correlation of predicted vs true counts.

    Calibration-free: a shrunk or shifted attacker still scores here if it
    ranks client-class pairs correctly. Reported alongside counts R² so a
    floor-level R² with positive Spearman reads as weak-but-real signal,
    not as no signal.
    """
    from scipy.stats import spearmanr

    left = np.concatenate([np.asarray(row, dtype=np.float64).reshape(-1) for row in counts])
    right = np.concatenate([np.asarray(row, dtype=np.float64).reshape(-1) for row in pred])
    if np.std(right) < 1e-12:
        return float("nan")
    coef = spearmanr(left, right).statistic
    return float(coef) if np.isfinite(coef) else float("nan")


def _metric_bundle(rows):
    from fedseismic.privacy.distribution import histogram_r2
    from fedseismic.privacy.mechanism import prior_presence_auc
    from fedseismic.privacy.meta_attacker import tpr_at_fpr

    pis = [row["pi"] for row in rows]
    counts = [row["counts"] for row in rows]
    pred = [row["pred_counts"] for row in rows]
    auc = _presence_auc(pis, pred)
    floor = prior_presence_auc(rows)
    labels = []
    values = []
    floor_values = []
    population = np.mean(np.stack(pis), axis=0)
    for row in rows:
        present = row["pi"] > ABSENT
        labels.extend(present.astype(int).tolist())
        values.extend(np.asarray(row["pred_counts"], dtype=np.float64).tolist())
        floor_values.extend(np.asarray(population, dtype=np.float64).tolist())
    return {
        "counts_r2": counts_r2(counts, pred),
        "counts_spearman": _counts_spearman(counts, pred),
        "counts_spearman_floor": _counts_spearman(
            counts, [np.mean(np.stack(counts), axis=0) for _ in counts]),
        "size_only_counts_r2": counts_r2(
            counts,
            [size * np.mean(np.stack(pis), axis=0)
             for size in [row["pred_size"] for row in rows]]),
        "size_r2": size_r2(counts, [row["pred_size"] for row in rows]),
        "prop_oracle_r2": proportions_r2(pis, [row["oracle"] for row in rows]),
        "prop_estimated_r2": proportions_r2(pis, [row["estimated"] for row in rows]),
        "support_auc": auc,
        "support_floor": floor,
        "support_leakage": auc - floor,
        "support_tpr": tpr_at_fpr(labels, values, fpr_cap=0.1),
        "support_tpr_floor": tpr_at_fpr(labels, floor_values, fpr_cap=0.1),
        "mix_r2": histogram_r2(pis, [to_simplex(np.clip(row, 0.0, None)) for row in pred]),
    }


def _load_shadow(run_dir):
    path = Path("results/attacks/shadows") / Path(run_dir).resolve().relative_to(Path("results/runs").resolve()) / "shadow.npz"
    if not path.is_file():
        raise FileNotFoundError(f"no shadow cache in {path}")
    with np.load(path) as handle:
        features = np.asarray(handle["features"], dtype=np.float64)
        mixes = np.asarray(handle["mixes"], dtype=np.float64)
        positive = np.asarray(handle["positive"], dtype=np.float64) if "positive" in handle.files else None
    return features, mixes, positive


def score_run(run_dir, targets, n_shadows=48):
    """Fit every attacker for one seed. Counts are checked against the saved mix."""
    from fedseismic.config import RunConfig
    from fedseismic.privacy.decaf import client_scores
    from fedseismic.privacy.head_channels import mean_channels
    from fedseismic.privacy.log import load_labels, load_meta

    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    seed = int(meta.get("seed", cfg.seed))
    histograms, _ = load_labels(run_dir)
    norms, bias, _participations = mean_channels(run_dir)
    decaf_rows = {row["client"]: row["score"] for row in client_scores(run_dir)}
    counts_by_client = real_counts(cfg, targets, seed)
    clients = sorted(client for client in bias if client in histograms and client < len(counts_by_client))
    pis = []
    counts = []
    for client in clients:
        count = counts_by_client[client]
        saved = to_simplex(histograms[client])
        rebuilt = to_simplex(count)
        if np.max(np.abs(saved - rebuilt)) > 1e-6:
            raise ValueError(f"rebuilt mix does not match labels for client {client} in {run_dir}")
        pis.append(saved)
        counts.append(count)
    features, mixes, positive = _load_shadow(run_dir)
    shadow = shadow_counts(cfg, targets, seed, n_shadows)
    if len(shadow) != len(features):
        raise ValueError(f"shadow count rows {len(shadow)} != features {len(features)} in {run_dir}")
    if np.max(np.abs(mixes - np.stack([to_simplex(row) for row in shadow]))) > 1e-6:
        raise ValueError(f"shadow counts do not match the saved mixes in {run_dir}")
    classes = cfg.num_classes
    real = {
        "bias": np.stack([bias[client] for client in clients]),
        "row": np.stack([norms[client] for client in clients]),
        "decaf": np.stack([decaf_rows[client] for client in clients]),
        "ridge": np.concatenate([
            np.stack([norms[client] for client in clients]),
            np.stack([bias[client] for client in clients]),
        ], axis=1),
    }
    shadow_channels = {
        "bias": features[:, classes:],
        "row": features[:, :classes],
        "decaf": positive,
    }
    grouped = {}
    for name, scores in shadow_channels.items():
        if scores is None:
            continue
        counts_hat, sizes_hat, support_hat = _channel_predictions(scores, shadow, real[name])
        grouped[name] = _rows_from_predictions(clients, pis, counts, counts_hat, sizes_hat, support_hat)
    counts_hat, sizes_hat, support_hat = _ridge_predictions(features, shadow, real["ridge"])
    grouped["ridge"] = _rows_from_predictions(clients, pis, counts, counts_hat, sizes_hat, support_hat)
    return grouped


def score_pair(run_dir, targets, n_shadows=48):
    """Row-size line and the pooled two-channel ridge, on the same clients."""
    from fedseismic.config import RunConfig
    from fedseismic.privacy.head_channels import mean_channels
    from fedseismic.privacy.log import load_labels, load_meta

    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    seed = int(meta.get("seed", cfg.seed))
    histograms, _ = load_labels(run_dir)
    norms, bias, _participations = mean_channels(run_dir)
    counts_by_client = real_counts(cfg, targets, seed)
    clients = sorted(client for client in bias if client in histograms and client < len(counts_by_client))
    pis = []
    counts = []
    for client in clients:
        count = counts_by_client[client]
        saved = to_simplex(histograms[client])
        rebuilt = to_simplex(count)
        if np.max(np.abs(saved - rebuilt)) > 1e-6:
            raise ValueError(f"rebuilt mix does not match labels for client {client} in {run_dir}")
        pis.append(saved)
        counts.append(count)
    features, mixes, _positive = _load_shadow(run_dir)
    shadow = shadow_counts(cfg, targets, seed, n_shadows)
    if len(shadow) != len(features):
        raise ValueError(f"shadow count rows {len(shadow)} != features {len(features)} in {run_dir}")
    if np.max(np.abs(mixes - np.stack([to_simplex(row) for row in shadow]))) > 1e-6:
        raise ValueError(f"shadow counts do not match the saved mixes in {run_dir}")
    classes = cfg.num_classes
    shadow_row = features[:, :classes]
    shadow_bias = features[:, classes:]
    real_row = np.stack([norms[client] for client in clients])
    real_bias = np.stack([bias[client] for client in clients])
    row_hat = _apply_line(_fit_line(shadow_row, shadow), real_row)
    scaler, model = _fit_pair(shadow_row, shadow_bias, shadow)
    pair_hat = _apply_pair(scaler, model, real_row, real_bias)
    coefficient = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    rows = []
    for index, client in enumerate(clients):
        rows.append({
            "client": int(client),
            "pi": np.asarray(pis[index], dtype=np.float64),
            "counts": np.asarray(counts[index], dtype=np.float64),
            "pred_counts": np.asarray(pair_hat[index], dtype=np.float64),
            "pred_row": np.asarray(row_hat[index], dtype=np.float64),
        })
    return rows, {
        "alpha": float(model.alpha_),
        "row_coef": float(coefficient[0]),
        "bias_coef": float(coefficient[1]),
    }


def _bootstrap_bundle(groups, n_draws, rng):
    from fedseismic.privacy.mechanism import _mean_finite, _percentile, _resample

    def mean_bundle(seeds):
        values = [_metric_bundle(rows) for rows in seeds]
        return {key: _mean_finite(item[key] for item in values) for key in values[0]}

    point = mean_bundle(groups)
    draws = [mean_bundle([_resample(group, rng) for group in groups]) for _ in range(n_draws)]
    bundled = {}
    for key in point:
        low, high = _percentile([item[key] for item in draws])
        bundled[key] = (point[key], low, high)
    return bundled


def summarize(grouped, n_draws=1000):
    """``grouped`` maps (dataset, smooth, attacker) to a list of per-seed row lists."""
    from fedseismic.privacy.mechanism import DISPLAY

    headlines = (
        "counts_r2", "counts_spearman", "size_only_counts_r2", "size_r2",
        "prop_oracle_r2", "prop_estimated_r2",
        "support_auc", "support_leakage", "support_tpr", "mix_r2",
    )
    zero_floor = {"counts_r2", "size_only_counts_r2", "size_r2", "prop_oracle_r2", "prop_estimated_r2", "mix_r2"}
    table = []
    rng = np.random.RandomState(0)
    for key in sorted(grouped):
        dataset, smooth, attacker = key
        print(f"intervals {DISPLAY.get(dataset, dataset)} λ={smooth:g} {attacker}", flush=True)
        bundled = _bootstrap_bundle(grouped[key], n_draws, rng)
        for metric in headlines:
            point, low, high = bundled[metric]
            if metric == "counts_spearman":
                floor = bundled["counts_spearman_floor"][0]
                leakage, leakage_low, leakage_high = point - floor, low - floor, high - floor
            elif metric in zero_floor:
                floor = 0.0
                leakage, leakage_low, leakage_high = point, low, high
            elif metric == "support_auc":
                floor = bundled["support_floor"][0]
                leakage, leakage_low, leakage_high = bundled["support_leakage"]
            elif metric == "support_tpr":
                floor = bundled["support_tpr_floor"][0]
                leakage = point - floor
                leakage_low = low - floor
                leakage_high = high - floor
            elif metric == "support_leakage":
                floor = 0.0
                leakage, leakage_low, leakage_high = point, low, high
            else:
                floor = float("nan")
                leakage = leakage_low = leakage_high = float("nan")
            table.append({
                "dataset": DISPLAY.get(dataset, dataset),
                "prior_smooth": smooth,
                "method": "FedLC" if abs(smooth) < 1e-9 else ("FedAvg" if abs(smooth - 1.0) < 1e-9 else f"λ={smooth:g}"),
                "attacker": attacker,
                "observer": OBSERVER,
                "metric": metric,
                "estimate": point,
                "low": low,
                "high": high,
                "floor": floor,
                "leakage": leakage,
                "leakage_low": leakage_low,
                "leakage_high": leakage_high,
                "mixes_support_and_proportions": metric == "mix_r2",
            })
    return table


SINGLE_CHANNELS = ("bias", "row", "decaf")
CHANNEL_COLOR = {"bias": "#7570b3", "row": "#d95f02", "decaf": "#1b9e77"}


def fedavg_ok(table, attacker, metric, minimum):
    checks = []
    for row in table:
        if row["attacker"] != attacker or row["metric"] != metric or row["method"] != "FedAvg":
            continue
        checks.append(row["estimate"] > minimum and row["low"] > (0.0 if metric != "support_auc" else row["floor"]))
    return bool(checks) and all(checks)


def channels_passing_counts(table):
    """Single-channel count attackers whose FedAvg interval sits above the floor."""
    return [name for name in SINGLE_CHANNELS if fedavg_ok(table, name, "counts_r2", 0.0)]


def best_count_row(table, dataset, method, prior_smooth, allowed=None):
    """The passing channel with the highest counts R² at this point."""
    if allowed is None:
        allowed = channels_passing_counts(table)
    allowed = set(allowed)
    rows = [
        row for row in table
        if row["dataset"] == dataset
        and row["method"] == method
        and abs(float(row["prior_smooth"]) - float(prior_smooth)) < 1e-9
        and row["metric"] == "counts_r2"
        and row["attacker"] in allowed
    ]
    if not rows:
        return None
    return max(rows, key=lambda row: row["estimate"])


def _bar(estimate, low, high):
    return [max(0.0, float(estimate) - float(low)), max(0.0, float(high) - float(estimate))]


def _label_offset(dataset, method):
    """Keep the Blood λ labels from landing on one another."""
    if dataset == "Blood" and method == "FedAvg":
        return (-52, 8)
    if dataset == "Blood" and method == "λ=0.3":
        return (5, -13)
    if dataset == "Blood" and method == "λ=0.1":
        return (5, 9)
    if dataset == "Blood" and method == "λ=0.02":
        return (5, -13)
    if dataset == "Blood" and method == "FedLC":
        return (4, 11)
    if method == "FedLC":
        return (5, -12)
    return (5, 8)


def plot_support_vs_counts(table, path, support_attacker="row"):
    """One panel per dataset.

    The x-axis is row-size support leakage. The y-axis is the counts R² of
    the best single channel that recovered FedAvg. The joint ridge is not
    drawn: it can score below the row-size line, and that is a property of
    the fit rather than of the counts.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    allowed = channels_passing_counts(table)
    datasets = ["Blood", "OrganC", "CIFAR-10"]
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 4.0), sharey=True)
    chosen = []
    y_high = 0.05
    for axis, dataset in zip(axes, datasets):
        support_rows = {}
        for row in table:
            if row["dataset"] != dataset or row["attacker"] != support_attacker or row["metric"] != "support_leakage":
                continue
            support_rows[(float(row["prior_smooth"]), row["method"])] = row
        points = []
        for smooth, method in sorted(support_rows):
            counts = best_count_row(table, dataset, method, smooth, allowed)
            if counts is None:
                continue
            support = support_rows[(smooth, method)]
            points.append((smooth, method, support, counts))
            chosen.append({
                "dataset": dataset,
                "prior_smooth": smooth,
                "method": method,
                "support_attacker": support_attacker,
                "counts_attacker": counts["attacker"],
                "support_leakage": support["estimate"],
                "counts_r2": counts["estimate"],
            })
        if not points:
            continue
        order = np.argsort([item[0] for item in points])
        xs = [points[index][2]["estimate"] for index in order]
        ys = [points[index][3]["estimate"] for index in order]
        axis.plot(xs, ys, color="0.45", linewidth=1, zorder=1)
        for smooth, method, support, counts in points:
            color = CHANNEL_COLOR.get(counts["attacker"], "black")
            axis.errorbar(
                support["estimate"], counts["estimate"],
                xerr=np.asarray([_bar(support["estimate"], support["low"], support["high"])]).T,
                yerr=np.asarray([_bar(counts["estimate"], counts["low"], counts["high"])]).T,
                fmt="o", color=color, capsize=2, zorder=2,
            )
            axis.annotate(
                f"{method}, {counts['attacker']}",
                (support["estimate"], counts["estimate"]),
                textcoords="offset points",
                xytext=_label_offset(dataset, method),
                fontsize=7, color=color,
            )
            y_high = max(y_high, float(counts["high"]), float(counts["estimate"]))
        axis.axhline(0.0, color="0.7", linewidth=0.8)
        axis.axvline(0.0, color="0.7", linewidth=0.8)
        axis.set_title(dataset)
        axis.set_xlabel("Support leakage, row size")
    axes[0].set_ylabel("Counts R², best channel")
    axes[0].set_ylim(-0.05, y_high + 0.08)
    used = [name for name in SINGLE_CHANNELS if any(row["counts_attacker"] == name for row in chosen)]
    if used:
        handles = [
            plt.Line2D([0], [0], marker="o", color=CHANNEL_COLOR[name], linestyle="None", label=name)
            for name in used
        ]
        axes[-1].legend(handles=handles, title="Counts", frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return chosen


def main():
    import pandas as pd

    from fedseismic.privacy.mechanism import iter_runs

    runs = iter_runs()
    cache = {}
    grouped = {}
    ordered = sorted(runs, key=lambda run: (0 if abs(run["prior_smooth"] - 1.0) < 1e-9 else 1, run["dataset"], run["prior_smooth"], run["seed"]))
    for run in ordered:
        print(f"{run['dataset']} λ={run['prior_smooth']:g} {run['seed']}", flush=True)
        if run["dataset"] not in cache:
            from fedseismic.config import RunConfig
            from fedseismic.privacy.log import load_meta
            cfg = RunConfig.from_mapping(load_meta(run["path"])["config"])
            cache[run["dataset"]] = _dataset_targets(cfg)
        scored = score_run(run["path"], cache[run["dataset"]])
        for attacker, rows in scored.items():
            grouped.setdefault((run["dataset"], run["prior_smooth"], attacker), []).append(rows)
            relative = Path(run["path"]).resolve().relative_to(Path("results/runs").resolve())
            saved = Path("results/attacks/howmuch") / relative / f"{attacker}.npz"
            saved.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                saved,
                client=np.asarray([row["client"] for row in rows], dtype=np.int32),
                counts=np.stack([row["counts"] for row in rows]),
                pred_counts=np.stack([row["pred_counts"] for row in rows]),
                pred_size=np.asarray([row["pred_size"] for row in rows]),
                oracle=np.stack([row["oracle"] for row in rows]),
                estimated=np.stack([row["estimated"] for row in rows]),
            )
    table = summarize(grouped)
    count_metrics = {"counts_r2", "counts_spearman", "size_r2", "prop_oracle_r2", "prop_estimated_r2", "mix_r2"}
    support_metrics = {"support_auc", "support_leakage", "support_tpr"}
    passed = fedavg_ok(table, "ridge", "counts_r2", 0.2) and fedavg_ok(table, "ridge", "support_auc", 0.0)
    for row in table:
        if row["method"] == "FedAvg":
            row["interpret"] = True
        elif row["metric"] in count_metrics:
            row["interpret"] = fedavg_ok(table, row["attacker"], "counts_r2", 0.0)
        elif row["metric"] in support_metrics:
            row["interpret"] = fedavg_ok(table, row["attacker"], "support_auc", 0.0)
        else:
            row["interpret"] = False
    frame = pd.DataFrame(table)
    path = Path("results/tables/howmuch.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    figure = Path("results/figures/support_vs_howmuch.png")
    plot_support_vs_counts(table, figure)
    print("FedAvg ridge control", "pass" if passed else "FAIL", flush=True)
    for row in table:
        if (row["attacker"] == "ridge"
                and row["metric"] in {"counts_r2", "support_auc", "size_r2"}
                and row["method"] in {"FedAvg", "FedLC"}):
            print(
                f"  {row['dataset']} {row['method']} {row['metric']} "
                f"{row['estimate']:.3f} [{row['low']:.3f}, {row['high']:.3f}]",
                flush=True,
            )
    print(f"wrote {path.resolve()}")
    print(f"wrote {figure.resolve()}")
    if not passed:
        print("STOP: the count regressor did not recover FedAvg. Other methods are not interpreted.")


def redraw_figure(csv_path="results/tables/howmuch.csv", figure_path="results/figures/support_vs_howmuch.png"):
    import pandas as pd

    frame = pd.read_csv(csv_path)
    chosen = plot_support_vs_counts(frame.to_dict("records"), figure_path)
    out = Path("results/tables/support_vs_howmuch_attackers.csv")
    pd.DataFrame(chosen).to_csv(out, index=False)
    print(f"wrote {Path(figure_path).resolve()}")
    print(f"wrote {out.resolve()}")
    for row in chosen:
        print(f"  {row['dataset']} {row['method']} counts={row['counts_attacker']}")
    return chosen


def diagnose_pair(n_draws=1000):
    """Compare the pooled two-channel ridge with the row-size line.

    The published joint ridge is left as it is. This fit is the one that
    contains the row-size line as a special case.
    """
    import pandas as pd

    from fedseismic.config import RunConfig
    from fedseismic.privacy.log import load_meta
    from fedseismic.privacy.mechanism import DISPLAY, _mean_finite, _percentile, _resample, iter_runs

    runs = iter_runs()
    cache = {}
    grouped = {}
    fitted = {}
    for run in sorted(runs, key=lambda item: (item["dataset"], item["prior_smooth"], item["seed"])):
        print(f"pair {run['dataset']} λ={run['prior_smooth']:g} {run['seed']}", flush=True)
        if run["dataset"] not in cache:
            cfg = RunConfig.from_mapping(load_meta(run["path"])["config"])
            cache[run["dataset"]] = _dataset_targets(cfg)
        rows, info = score_pair(run["path"], cache[run["dataset"]])
        key = (run["dataset"], run["prior_smooth"])
        grouped.setdefault(key, []).append(rows)
        fitted.setdefault(key, []).append(info)
    rng = np.random.RandomState(0)
    table = []
    for dataset, smooth in sorted(grouped):
        groups = grouped[(dataset, smooth)]

        def mean_stats(seeds):
            pair = _mean_finite(
                counts_r2([row["counts"] for row in rows], [row["pred_counts"] for row in rows]) for rows in seeds
            )
            line = _mean_finite(
                counts_r2([row["counts"] for row in rows], [row["pred_row"] for row in rows]) for rows in seeds
            )
            return pair, line, pair - line

        point = mean_stats(groups)
        draws = [mean_stats([_resample(group, rng) for group in groups]) for _ in range(n_draws)]
        lows, highs = [], []
        for index in range(3):
            low, high = _percentile([item[index] for item in draws])
            lows.append(low)
            highs.append(high)
        info = fitted[(dataset, smooth)]
        method = "FedLC" if abs(smooth) < 1e-9 else ("FedAvg" if abs(smooth - 1.0) < 1e-9 else f"λ={smooth:g}")
        table.append({
            "dataset": DISPLAY.get(dataset, dataset),
            "prior_smooth": smooth,
            "method": method,
            "pair_r2": point[0],
            "pair_low": lows[0],
            "pair_high": highs[0],
            "row_r2": point[1],
            "row_low": lows[1],
            "row_high": highs[1],
            "gap": point[2],
            "gap_low": lows[2],
            "gap_high": highs[2],
            "alpha_median": float(np.median([item["alpha"] for item in info])),
            "row_coef_mean": float(np.mean([item["row_coef"] for item in info])),
            "bias_coef_mean": float(np.mean([item["bias_coef"] for item in info])),
        })
        print(
            f"  {DISPLAY.get(dataset, dataset)} {method} pair {point[0]:.3f} "
            f"[{lows[0]:.3f}, {highs[0]:.3f}] row {point[1]:.3f} gap {point[2]:+.3f} "
            f"[{lows[2]:+.3f}, {highs[2]:+.3f}]",
            flush=True,
        )
    frame = pd.DataFrame(table)
    path = Path("results/tables/pair_ridge.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {path.resolve()}")
    return table


def _method_label(cfg):
    if getattr(cfg, "local_steps", None):
        balance = "balanced" if str(cfg.gradient_balance) == "class" else "count"
        return f"{balance}-{int(cfg.local_steps)}"
    if str(getattr(cfg, "gradient_balance", "example")) == "class_minibatch":
        tag = "balanced-mb"
        if getattr(cfg, "balanced_steps", None):
            tag = f"{tag}-t{int(cfg.balanced_steps)}"
        return tag
    if str(cfg.algorithm) == "fedkper" and bool(cfg.logit_adjust):
        return "FedKPer+logit"
    if str(cfg.algorithm) == "fedkper":
        return "FedKPer"
    if bool(cfg.logit_adjust) and abs(float(cfg.prior_smooth)) < 1e-9:
        return "FedLC"
    if abs(float(cfg.prior_smooth) - 1.0) < 1e-9:
        return "FedAvg"
    return f"λ={float(cfg.prior_smooth):g}"


def score_saved_runs(roots, n_draws=1000, output=None):
    """Score finished runs whose shadows use the victim's trainer.

    The FedAvg control for these attackers is the one already in
    ``results/tables/howmuch.csv``. This table does not repeat it.
    ``output`` defaults to ``results/tables/fedkper_howmuch.csv`` for the
    FedKPer rerun; pass an explicit path for any other run family so tables
    are never overwritten.
    """
    import pandas as pd
    import torch

    from fedseismic.config import RunConfig, resolve_device
    from fedseismic.privacy.log import load_meta
    from fedseismic.privacy.mechanism import DISPLAY
    from fedseismic.privacy.share_sweep import _load_shadows

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    found = []
    for root in roots:
        for run in sorted(Path(root).glob("*/seed_*")):
            if (run / "results.json").is_file() and (run / "head").is_dir():
                found.append(run)
    if not found:
        raise FileNotFoundError(f"no finished runs under {list(roots)}")
    cache = {}
    grouped = {}
    for run in found:
        meta = load_meta(run)
        cfg = RunConfig.from_mapping(meta["config"])
        print(f"shadows {run}", flush=True)
        _load_shadows(run, cfg, device, 48)
        if cfg.dataset not in cache:
            cache[cfg.dataset] = _dataset_targets(cfg)
        print(f"score {run}", flush=True)
        scored = score_run(run, cache[cfg.dataset])
        method = _method_label(cfg)
        relative = Path(run).resolve().relative_to(Path("results/runs").resolve())
        for attacker, rows in scored.items():
            grouped.setdefault((cfg.dataset, method, attacker), []).append(rows)
            saved = Path("results/attacks/howmuch") / relative / f"{attacker}.npz"
            saved.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                saved,
                client=np.asarray([row["client"] for row in rows], dtype=np.int32),
                counts=np.stack([row["counts"] for row in rows]),
                pred_counts=np.stack([row["pred_counts"] for row in rows]),
                pred_size=np.asarray([row["pred_size"] for row in rows]),
                oracle=np.stack([row["oracle"] for row in rows]),
                estimated=np.stack([row["estimated"] for row in rows]),
            )
    headlines = ("counts_r2", "counts_spearman", "size_only_counts_r2", "size_r2", "support_auc", "support_leakage", "support_tpr",
                 "prop_oracle_r2", "prop_estimated_r2")
    rng = np.random.RandomState(0)
    table = []
    for dataset, method, attacker in sorted(grouped):
        print(f"intervals {method} {attacker}", flush=True)
        bundled = _bootstrap_bundle(grouped[(dataset, method, attacker)], n_draws, rng)
        for metric in headlines:
            point, low, high = bundled[metric]
            if metric == "counts_spearman":
                floor = bundled["counts_spearman_floor"][0]
            elif metric == "support_auc":
                floor = bundled["support_floor"][0]
            elif metric == "support_tpr":
                floor = bundled["support_tpr_floor"][0]
            else:
                floor = 0.0
            if metric == "support_leakage":
                leakage, leakage_low, leakage_high = point, low, high
            elif metric == "support_auc":
                leakage, leakage_low, leakage_high = bundled["support_leakage"]
            else:
                leakage, leakage_low, leakage_high = point, low, high
            table.append({
                "dataset": DISPLAY.get(dataset, dataset),
                "method": method,
                "attacker": attacker,
                "observer": OBSERVER,
                "metric": metric,
                "estimate": point,
                "low": low,
                "high": high,
                "floor": floor,
            })
            if (attacker in {"row", "decaf", "bias"} and metric in {
                    "counts_r2", "counts_spearman", "size_only_counts_r2",
                    "support_auc", "support_tpr", "prop_estimated_r2"}):
                print(f"  {method} {attacker} {metric} {point:.3f} [{low:.3f}, {high:.3f}]", flush=True)
    frame = pd.DataFrame(table)
    path = Path(output or "results/tables/fedkper_howmuch.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {path.resolve()}")
    return table


def main_cli(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Counts and support rescoring")
    parser.add_argument("--figure", action="store_true", help="redraw the figure from howmuch.csv")
    parser.add_argument("--pair", action="store_true", help="refit the pooled two-channel ridge")
    parser.add_argument("--extra", nargs="+", help="score these finished run roots")
    parser.add_argument("--output", default=None, help="output CSV for --extra (default results/tables/fedkper_howmuch.csv)")
    args = parser.parse_args(argv)
    if args.figure:
        return redraw_figure()
    if args.pair:
        return diagnose_pair()
    if args.extra:
        return score_saved_runs(args.extra, output=args.output)
    return main()


if __name__ == "__main__":
    main_cli()
