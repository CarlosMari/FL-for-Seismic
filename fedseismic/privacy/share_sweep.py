"""Learned attacker on absent-versus-share bins.

Shadows train with the victim's loss from a received model saved in that run.
The fit never sees a real client's mix. FedAvg (λ = 1) is scored first. If that
positive control misses, later methods for that dataset are not scored.
"""

from pathlib import Path
import json

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from fedseismic.config import RunConfig, resolve_device
from fedseismic.privacy.log import _load_checkpoint, load_labels
from fedseismic.privacy.meta_attacker import (
    SHARE_BINS,
    _predict_client,
    _presence_model,
    _presence_scores,
    _summarize,
    collect_shadows,
    participation_features,
    share_bin_metrics,
    training_pool,
)
from fedseismic.privacy.metrics import to_simplex
from fedseismic.privacy.prior_curve import ROOTS, _runs

ATTACKS = Path("results/attacks")


def _cache_path(run_dir):
    relative = Path(run_dir).resolve().relative_to(Path("results/runs").resolve())
    return ATTACKS / "shadows" / relative / "shadow.npz"


def _prediction_path(run_dir):
    relative = Path(run_dir).resolve().relative_to(Path("results/runs").resolve())
    return ATTACKS / "predictions" / relative / "learned.npz"


def _received_states(run_dir):
    paths = sorted((Path(run_dir) / "received").glob("round_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no received models in {run_dir}")
    return [_load_checkpoint(path) for path in paths]


def _load_shadows(run_dir, cfg, device, n_shadows, cohort_size=None, start_round_max=None):
    path = _cache_path(run_dir)
    if path.is_file():
        with np.load(path) as handle:
            def _cached_int(name):
                import numpy as _np
                return int(_np.asarray(handle[name]).reshape(-1)[0]) if name in handle.files else 0
            cached_cohort = _cached_int("cohort_size")
            want_cohort = int(cohort_size) if cohort_size else 0
            cached_start = _cached_int("start_round_max")
            want_start = int(start_round_max) if start_round_max else 0
            if (len(handle["features"]) >= int(n_shadows) and cached_cohort == want_cohort
                    and cached_start == want_start):
                positive = handle["positive"] if "positive" in handle.files else None
                return handle["features"], handle["mixes"], positive
            print(f"  stale cache {path} has {len(handle['features'])} (cohort {cached_cohort}, "
                  f"startmax {cached_start}), want {n_shadows} (cohort {want_cohort}, "
                  f"startmax {want_start}), regenerating", flush=True)
    from fedseismic.experiment import build_federated_run

    meta = json.loads((Path(run_dir) / "run_meta.json").read_text())
    seed = int(meta.get("seed", cfg.seed))
    data = build_federated_run(cfg, seed)
    images, targets, n_channels = training_pool(cfg)
    received = _received_states(run_dir)
    features, mixes, positive = collect_shadows(
        cfg, received[0], data.model_factory, images, targets,
        n_channels, None, device, n_shadows, seed,
        start_states=received, return_positive=True, cohort_size=cohort_size,
        start_round_max=start_round_max,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features, mixes=mixes, positive=positive,
                        cohort_size=np.asarray([int(cohort_size) if cohort_size else 0]),
                        start_round_max=np.asarray([int(start_round_max) if start_round_max else 0]))
    return features, mixes, positive


def _save_predictions(run_dir, clients, truths, mix_hat, presence_hat, scaler, mix_model, presence_models):
    participations = []
    for path in sorted((Path(run_dir) / "head").glob("round_*_client_*.npz")):
        with np.load(path) as handle:
            weight = np.asarray(handle["weight_delta"], dtype=np.float64)
            bias = np.asarray(handle["bias_delta"], dtype=np.float64).reshape(-1)
        sizes = np.linalg.norm(weight.reshape(weight.shape[0], -1), axis=1)
        feature = np.concatenate([sizes, bias])
        parts = path.stem.split("_")
        participations.append((int(parts[1]), int(parts[-1]), feature))
    matrix = scaler.transform(np.stack([row[2] for row in participations]))
    mix = mix_model.predict(matrix)
    presence = np.zeros_like(mix)
    for class_index, classifier in enumerate(presence_models):
        if classifier is None:
            presence[:, class_index] = 0.5
        else:
            presence[:, class_index] = classifier.predict_proba(matrix)[:, 1]
    path = _prediction_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        client=np.asarray(clients, dtype=np.int32),
        truth=np.stack(truths),
        mix_hat=np.stack(mix_hat),
        presence_hat=np.stack(presence_hat),
        participation_round=np.asarray([row[0] for row in participations], dtype=np.int32),
        participation_client=np.asarray([row[1] for row in participations], dtype=np.int32),
        participation_mix_hat=mix,
        participation_presence_hat=presence,
    )


def score_run(run_dir, device, n_shadows=48):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    cfg = RunConfig.from_mapping(meta["config"])
    histograms, _ = load_labels(run_dir)
    grouped = participation_features(run_dir)
    clients = sorted(client for client in grouped if client in histograms)
    truths = [to_simplex(histograms[client]) for client in clients]
    means = np.stack([np.mean(grouped[client], axis=0) for client in clients])
    shadow_x, shadow_y, _positive = _load_shadows(run_dir, cfg, device, n_shadows)
    scaler = StandardScaler().fit(shadow_x)
    scaled = scaler.transform(shadow_x)
    mix_model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(scaled, shadow_y)
    mix_hat = [_predict_client(mix_model, scaler, grouped[client]) for client in clients]
    presence_models = _presence_model(scaled, shadow_y)
    presence_hat = _presence_scores(presence_models, scaler, grouped, clients)
    _save_predictions(run_dir, clients, truths, mix_hat, presence_hat, scaler, mix_model, presence_models)
    real_mean = scaler.transform(means)
    real = np.stack(truths)
    bins = []
    for label, lo, hi in SHARE_BINS:
        scored = share_bin_metrics(scaled, shadow_y, real_mean, real, lo, hi)
        bins.append({"bin": label, **scored})
    summary = _summarize(truths, mix_hat)
    return {
        "dataset": str(cfg.dataset),
        "prior_smooth": float(cfg.prior_smooth),
        "seed": run_dir.name,
        "n_shadows": int(len(shadow_x)),
        "mix_pearson": summary["pearson"],
        "mix_loo_r2": summary["loo_r2"],
        "mix_presence_auc": summary["presence_auc"],
        "bins": bins,
    }


def _display(dataset):
    return {"bloodmnist": "Blood", "organcmnist": "OrganC", "cifar10": "CIFAR-10"}.get(dataset, dataset)


def _method(smooth):
    if smooth == 0.0:
        return "FedLC"
    if smooth == 1.0:
        return "FedAvg"
    return f"λ={smooth:g}"


def fedavg_recovered(rows):
    """True when the learned mix on ordinary training is above the floor."""
    if not rows:
        return False
    pearson = float(np.mean([row["mix_pearson"] for row in rows]))
    loo = float(np.mean([row["mix_loo_r2"] for row in rows]))
    return pearson > 0.4 and loo > 0.15


def _ordered_runs():
    found = []
    for root in ROOTS:
        if not root.is_dir():
            continue
        for run in _runs(root):
            meta = json.loads((run / "run_meta.json").read_text())
            found.append((float(meta["config"]["prior_smooth"]), str(meta["config"]["dataset"]), run))
    found.sort(key=lambda item: (0 if item[0] == 1.0 else 1, item[1], item[0], str(item[2])))
    return found


def _real_features(run_dir):
    grouped = participation_features(run_dir)
    histograms, _ = load_labels(run_dir)
    clients = sorted(client for client in grouped if client in histograms)
    means = np.stack([np.mean(grouped[client], axis=0) for client in clients])
    truths = np.stack([to_simplex(histograms[client]) for client in clients])
    return means, truths


def _prepared_seeds():
    """Shadow features and real client matrices for the FedAvg and FedLC endpoints."""
    from sklearn.preprocessing import StandardScaler

    grouped = {}
    for smooth, dataset, run in _ordered_runs():
        if abs(smooth) > 1e-9 and abs(smooth - 1.0) > 1e-9:
            continue
        cache = _cache_path(run)
        if not cache.is_file():
            continue
        with np.load(cache) as handle:
            features = np.asarray(handle["features"], dtype=np.float64)
            mixes = np.asarray(handle["mixes"], dtype=np.float64)
        real, truths = _real_features(run)
        scaler = StandardScaler().fit(features)
        grouped.setdefault((dataset, smooth), []).append({
            "scaled": scaler.transform(features),
            "mixes": mixes,
            "real": scaler.transform(real),
            "truths": truths,
        })
    return grouped


def _mean_score(seeds, scorer, lo, hi, index_of=None):
    accuracies = []
    aucs = []
    absent = 0
    inside = 0
    for seed in seeds:
        if index_of is None:
            real = seed["real"]
            truths = seed["truths"]
        else:
            chosen = index_of(seed)
            real = seed["real"][chosen]
            truths = seed["truths"][chosen]
        scored = scorer(seed["scaled"], seed["mixes"], real, truths, lo, hi)
        accuracies.append(scored["accuracy"])
        aucs.append(scored["auc"])
        absent += int(scored.get("n_real_absent") or 0)
        inside += int(scored.get("n_real_bin") or 0)
    return {
        "accuracy": float(np.nanmean(accuracies)),
        "auc": float(np.nanmean(aucs)),
        "n_real_absent": absent,
        "n_real_bin": inside,
    }


def bin_intervals(n_draws=1000):
    """Mean across seeds. Each draw resamples clients inside every seed."""
    import pandas as pd

    from fedseismic.privacy.meta_attacker import pooled_bin_metrics

    grouped = _prepared_seeds()
    rows = []
    rng = np.random.RandomState(0)
    for (dataset, smooth), seeds in sorted(grouped.items()):
        print(f"intervals {_display(dataset)} {_method(smooth)}", flush=True)
        for label, lo, hi in SHARE_BINS:
            for attacker, scorer in (
                ("learned", share_bin_metrics),
                ("learned_pooled", pooled_bin_metrics),
            ):
                point = _mean_score(seeds, scorer, lo, hi)
                drawn_acc = []
                drawn_auc = []
                for _ in range(n_draws):
                    def index_of(seed, _rng=rng):
                        n = len(seed["truths"])
                        return _rng.randint(0, n, size=n)
                    scored = _mean_score(seeds, scorer, lo, hi, index_of=index_of)
                    drawn_acc.append(scored["accuracy"])
                    drawn_auc.append(scored["auc"])
                acc = np.asarray(drawn_acc, dtype=np.float64)
                auc = np.asarray(drawn_auc, dtype=np.float64)
                rows.append({
                    "dataset": _display(dataset),
                    "method": _method(smooth),
                    "attacker": attacker,
                    "bin": label,
                    "accuracy": point["accuracy"],
                    "accuracy_low": float(np.nanpercentile(acc, 2.5)),
                    "accuracy_high": float(np.nanpercentile(acc, 97.5)),
                    "auc": point["auc"],
                    "auc_low": float(np.nanpercentile(auc, 2.5)),
                    "auc_high": float(np.nanpercentile(auc, 97.5)),
                    "n_real_absent": point["n_real_absent"],
                    "n_real_bin": point["n_real_bin"],
                })
                print(
                    f"  {attacker} {label} acc {point['accuracy']:.3f} auc {point['auc']:.3f}",
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    path = Path("results/tables/share_sweep_intervals.csv")
    frame.to_csv(path, index=False)
    print(f"wrote {path.resolve()}")


def main():
    import pandas as pd

    device = resolve_device("cuda")
    rows = []
    ordered = _ordered_runs()
    expected = {}
    for smooth, dataset, _run in ordered:
        if abs(smooth - 1.0) < 1e-9:
            expected[dataset] = expected.get(dataset, 0) + 1
    blocked = set()
    fedavg = {}
    for smooth, dataset, run in ordered:
        if dataset in blocked and abs(smooth - 1.0) > 1e-9:
            print(f"skip {dataset} λ={smooth:g} {run.name}: FedAvg control failed", flush=True)
            continue
        print(f"{dataset} λ={smooth:g} {run.name}", flush=True)
        scored = score_run(run, device)
        fedavg.setdefault(dataset, [])
        if abs(smooth - 1.0) < 1e-9:
            fedavg[dataset].append(scored)
        for item in scored["bins"]:
            rows.append({
                "dataset": _display(dataset),
                "method": _method(smooth),
                "prior_smooth": smooth,
                "seed": scored["seed"],
                "attacker": "learned",
                "bin": item["bin"],
                "accuracy": item["accuracy"],
                "auc": item["auc"],
                "n_classes": item["n_classes"],
                "n_real_absent": item["n_real_absent"],
                "n_real_bin": item["n_real_bin"],
                "mix_pearson": scored["mix_pearson"],
                "mix_loo_r2": scored["mix_loo_r2"],
                "mix_presence_auc": scored["mix_presence_auc"],
                "n_shadows": scored["n_shadows"],
            })
        print(
            f"  mix pearson {scored['mix_pearson']:+.3f} loo {scored['mix_loo_r2']:+.3f} "
            f"bins " + ", ".join(
                f"{item['bin']} acc {item['accuracy']:.3f} auc {item['auc']:.3f}" for item in scored["bins"]
            ),
            flush=True,
        )
        if abs(smooth - 1.0) < 1e-9 and len(fedavg[dataset]) == expected[dataset]:
            ok = fedavg_recovered(fedavg[dataset])
            print(f"FedAvg control {dataset}: {'pass' if ok else 'FAIL'}", flush=True)
            if not ok:
                blocked.add(dataset)
    frame = pd.DataFrame(rows)
    output = Path("results/tables/share_sweep.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    if len(frame):
        print(frame.groupby(["dataset", "method", "bin"])[["accuracy", "auc"]].mean().to_string())
    print(f"wrote {output.resolve()}")
    if blocked:
        print("STOP: FedAvg was not recovered for", ", ".join(sorted(blocked)))


if __name__ == "__main__":
    import sys
    if "--intervals" in sys.argv:
        bin_intervals()
    else:
        main()
