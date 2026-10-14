"""Curve from FedLC to FedAvg by smoothing the training prior.

π′ = (1−λ)π + λ/K. λ = 0 is FedLC. λ = 1 is a constant prior, which is FedAvg.
Shadows for the learned attacker each start from a received model saved during
the run, not from the final global model.
"""

from pathlib import Path
import json

import numpy as np

from fedseismic.config import RunConfig, resolve_device
from fedseismic.privacy.head_channels import _percentile_interval
from fedseismic.privacy.log import _load_checkpoint, load_labels
from fedseismic.privacy.meta_attacker import (
    HIGH_SHARE, collect_shadows, participation_features, share_accuracy, training_pool,
    _pooled_tpr, _predict_client, _presence_model, _presence_scores, _summarize,
)
from fedseismic.privacy.row_delta import _pearson, _presence_auc
from fedseismic.privacy.metrics import to_simplex
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOTS = (
    Path("results/runs/prior_smooth_blood"),
    Path("results/runs/prior_smooth_organc"),
    Path("results/runs/prior_smooth_cifar"),
)
SHARE_LEVELS = (0.01, 0.05, 0.10, 0.25)


def _runs(root):
    found = []
    for job in sorted(path for path in root.iterdir() if path.is_dir()):
        for run in sorted(job.glob("seed_*")):
            if (run / "results.json").is_file():
                found.append(run)
    if not found:
        raise FileNotFoundError(f"no finished runs under {root}")
    return found


def _with_interval(summary, truths, scores, n_draws=1000):
    rng = np.random.RandomState(0)
    pearson = []
    auc = []
    n_clients = len(truths)
    for _ in range(n_draws):
        index = rng.randint(0, n_clients, size=n_clients)
        sample_truth = [truths[item] for item in index]
        sample_score = [scores[item] for item in index]
        pearson.append(_pearson(sample_truth, sample_score))
        auc.append(_presence_auc(sample_truth, sample_score))
    low_p, high_p = _percentile_interval(pearson)
    low_a, high_a = _percentile_interval(auc)
    scored = dict(summary)
    scored["pearson_low"] = low_p
    scored["pearson_high"] = high_p
    scored["presence_auc_low"] = low_a
    scored["presence_auc_high"] = high_a
    return scored


def _received_states(run_dir):
    paths = sorted((Path(run_dir) / "received").glob("round_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no received models in {run_dir}")
    return [_load_checkpoint(path) for path in paths]


def score_run(run_dir, device, n_shadows=48):
    from fedseismic.experiment import build_federated_run

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    cfg = RunConfig.from_mapping(meta["config"])
    seed = int(meta.get("seed", cfg.seed))
    data = build_federated_run(cfg, seed)
    images, targets, n_channels = training_pool(cfg)
    histograms, _ = load_labels(run_dir)
    grouped = participation_features(run_dir)
    clients = sorted(client for client in grouped if client in histograms)
    truths = [to_simplex(histograms[client]) for client in clients]
    means = np.stack([np.mean(grouped[client], axis=0) for client in clients])
    bias_scores = [row[cfg.num_classes:] for row in means]
    row_scores = [row[:cfg.num_classes] for row in means]
    hand_bias = _with_interval(_summarize(truths, bias_scores), truths, bias_scores)
    hand_row = _with_interval(_summarize(truths, row_scores), truths, row_scores)

    received = _received_states(run_dir)
    shadow_x, shadow_y = collect_shadows(
        cfg, received[0], data.model_factory, images, targets,
        n_channels, None, device, n_shadows, seed,
        start_states=received,
    )
    scaler = StandardScaler().fit(shadow_x)
    scaled = scaler.transform(shadow_x)
    mix_model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(scaled, shadow_y)
    mix_hat = [_predict_client(mix_model, scaler, grouped[client]) for client in clients]
    presence_hat = _presence_scores(_presence_model(scaled, shadow_y), scaler, grouped, clients)
    real_mean = scaler.transform(means)
    shares = {
        f"share_{level:g}": share_accuracy(scaled, shadow_y, real_mean, np.stack(truths), high=level)
        for level in SHARE_LEVELS
    }
    local = float(json.loads((run_dir / "results.json").read_text())["local_mean"])
    return {
        "dataset": str(cfg.dataset),
        "prior_smooth": float(cfg.prior_smooth),
        "local_mean": local,
        "hand_bias": hand_bias,
        "hand_row": hand_row,
        "learned_mix": _with_interval(_summarize(truths, mix_hat), truths, mix_hat),
        "learned_presence": _with_interval({
            **_summarize(truths, presence_hat),
            "tpr_at_10": _pooled_tpr(truths, presence_hat),
            "share_acc": shares[f"share_{HIGH_SHARE:g}"],
            **shares,
        }, truths, presence_hat),
    }


def main():
    import pandas as pd

    device = resolve_device("cuda")
    rows = []
    for root in ROOTS:
        for run in _runs(root):
            print(run, flush=True)
            scored = score_run(run, device)
            dataset = scored.pop("dataset")
            smooth = scored.pop("prior_smooth")
            local = scored.pop("local_mean")
            for attack, summary in scored.items():
                row = {
                    "dataset": dataset,
                    "prior_smooth": smooth,
                    "seed": run.name,
                    "attack": attack,
                    "local_mean": local,
                    "pearson": summary["pearson"],
                    "presence_auc": summary["presence_auc"],
                    "pearson_low": summary.get("pearson_low", float("nan")),
                    "pearson_high": summary.get("pearson_high", float("nan")),
                    "presence_auc_low": summary.get("presence_auc_low", float("nan")),
                    "presence_auc_high": summary.get("presence_auc_high", float("nan")),
                    "loo_r2": summary["loo_r2"],
                    "tpr_at_10": summary.get("tpr_at_10", float("nan")),
                    "share_acc": summary.get("share_acc", float("nan")),
                }
                for level in SHARE_LEVELS:
                    row[f"share_{level:g}"] = summary.get(f"share_{level:g}", float("nan"))
                rows.append(row)
                print(
                    f"  {dataset} λ={smooth:g} {attack} pearson {summary['pearson']:+.3f} "
                    f"auc {summary['presence_auc']:.3f} "
                    f"[{summary.get('presence_auc_low', float('nan')):.3f}, "
                    f"{summary.get('presence_auc_high', float('nan')):.3f}] "
                    f"local {local:.3f}",
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    output = Path("results/tables/prior_smooth.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    blood = frame[frame["dataset"] == "bloodmnist"]
    blood.to_csv(Path("results/tables/prior_smooth_blood.csv"), index=False)
    print(frame.groupby(["dataset", "prior_smooth", "attack"])[
        ["pearson", "presence_auc", "loo_r2", "tpr_at_10", "share_acc", "local_mean"]
    ].mean().to_string())
    print(f"wrote {output.resolve()}")


if __name__ == "__main__":
    main()
