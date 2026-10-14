"""Per-class shift of public feature means, received body against the upload.

The line from that shift to the class count is fit on shadow clients that
train with the victim's loss from a received model. FedAvg has to clear the
counts floor on every dataset before any other method is read.
"""

from pathlib import Path

import numpy as np
import torch

from fedseismic.privacy.howmuch import _apply_line, _fit_line, counts_r2, real_counts
from fedseismic.privacy.metrics import to_simplex
from fedseismic.privacy.row_delta import _class_means, _feature_table, body_shift

ABSENT = 1e-3
FEDAVG_ROOTS = {
    "Blood": "results/runs/prior_smooth_blood/fedavg_noniid_a0.1_lNA_dna_p1",
    "OrganC": "results/runs/prior_smooth_organc/fedavg_noniid_a0.1_lNA_dna_p1",
    "CIFAR-10": "results/runs/prior_smooth_cifar/fedavg_noniid_a0.1_lNA_dna_p1",
}


def public_shift(factory, received_state, upload_state, loader, device, num_classes):
    """L2 move of each class's mean public feature, upload minus received."""
    from fedseismic.privacy.estimators import clone_model_from_state

    received = clone_model_from_state(factory, received_state, device=device)
    uploaded = clone_model_from_state(factory, upload_state, device=device)
    received_features, received_labels = _feature_table(received, loader, device)
    upload_features, upload_labels = _feature_table(uploaded, loader, device)
    if not np.array_equal(received_labels, upload_labels):
        raise ValueError("the public probe changed labels between the two bodies")
    _overall, received_means, _counts = _class_means(received_features, received_labels, num_classes)
    _overall, upload_means, _counts = _class_means(upload_features, upload_labels, num_classes)
    return body_shift(upload_means, received_means)


def _cache_path(run_dir):
    relative = Path(run_dir).resolve().relative_to(Path("results/runs").resolve())
    return Path("results/attacks/feature_shift") / relative / "shift.npz"


def _loader_counts(loader, num_classes):
    labels = np.asarray(loader.dataset.targets)[list(loader.dataset.indices)]
    return np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64)


def _collect(run_dir, device, n_shadows=48):
    from fedseismic.config import RunConfig
    from fedseismic.experiment import build_federated_run
    from fedseismic.privacy.body_channel import last_rounds
    from fedseismic.privacy.log import _load_checkpoint, load_labels, load_local_states, load_meta
    from fedseismic.privacy.meta_attacker import (
        _shadow_loaders,
        make_shadow_trainer,
        training_pool,
    )
    from fedseismic.privacy.share_sweep import _received_states

    path = _cache_path(run_dir)
    if path.is_file():
        with np.load(path) as handle:
            return (
                np.asarray(handle["client"]),
                np.asarray(handle["counts"]),
                np.asarray(handle["shift"]),
                np.asarray(handle["shadow_counts"]),
                np.asarray(handle["shadow_shift"]),
            )
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    seed = int(meta.get("seed", cfg.seed))
    data = build_federated_run(cfg, seed)
    histograms, _ = load_labels(run_dir)
    local_states = load_local_states(run_dir)
    rounds = last_rounds(run_dir)
    targets = real_counts(cfg, training_pool(cfg)[1], seed)
    clients = []
    counts = []
    shifts = []
    for client, state in sorted(local_states.items()):
        if client not in histograms or client >= len(targets):
            continue
        received_path = run_dir / "received" / f"round_{rounds.get(client, 0):03d}.pt"
        if not received_path.is_file():
            continue
        saved = to_simplex(histograms[client])
        rebuilt = to_simplex(targets[client])
        if np.max(np.abs(saved - rebuilt)) > 1e-6:
            raise ValueError(f"rebuilt mix does not match labels for client {client} in {run_dir}")
        shift = public_shift(
            data.model_factory, _load_checkpoint(received_path), state,
            data.probe_loader, device, cfg.num_classes,
        )
        clients.append(client)
        counts.append(targets[client])
        shifts.append(shift)
        print(f"  client {client}", flush=True)
    if not clients:
        raise FileNotFoundError(f"no received bodies in {run_dir}")
    images, pool_targets, n_channels = training_pool(cfg)
    received_states = _received_states(run_dir)
    pick = np.random.RandomState(40_000 + seed)
    shadow_shifts = []
    shadow_counts = []
    for index, loader in enumerate(_shadow_loaders(cfg, images, pool_targets, n_channels, n_shadows, seed)):
        state = received_states[int(pick.randint(len(received_states)))]
        trainer = make_shadow_trainer(cfg, data.model_factory(), loader, device)
        trainer.download(state)
        torch.manual_seed(50_000 + seed + index)
        trainer.train(global_state=state)
        shadow_shifts.append(public_shift(
            data.model_factory, state, trainer.upload(), data.probe_loader, device, cfg.num_classes,
        ))
        shadow_counts.append(_loader_counts(loader, cfg.num_classes))
        print(f"  shadow {index + 1}/{n_shadows}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        client=np.asarray(clients, dtype=np.int32),
        counts=np.stack(counts),
        shift=np.stack(shifts),
        shadow_counts=np.stack(shadow_counts),
        shadow_shift=np.stack(shadow_shifts),
    )
    return (
        np.asarray(clients),
        np.stack(counts),
        np.stack(shifts),
        np.stack(shadow_counts),
        np.stack(shadow_shifts),
    )


def _predict(shadow_shift, shadow_counts, shift):
    return _apply_line(_fit_line(shadow_shift, shadow_counts), shift)


def _bundle(rows):
    from fedseismic.privacy.mechanism import prior_presence_auc
    from fedseismic.privacy.row_delta import _presence_auc

    counts = [row["counts"] for row in rows]
    pred = [row["pred"] for row in rows]
    pis = [row["pi"] for row in rows]
    auc = _presence_auc(pis, pred)
    floor = prior_presence_auc([{"pi": row["pi"]} for row in rows])
    return {
        "counts_r2": counts_r2(counts, pred),
        "support_auc": auc,
        "support_leakage": auc - floor,
    }


def summarize(grouped, n_draws=1000):
    from fedseismic.privacy.mechanism import _mean_finite, _percentile, _resample

    rng = np.random.RandomState(0)
    table = []
    for dataset in sorted(grouped):
        groups = grouped[dataset]

        def mean_stats(seeds):
            values = [_bundle(rows) for rows in seeds]
            return {key: _mean_finite(item[key] for item in values) for key in values[0]}

        point = mean_stats(groups)
        draws = [mean_stats([_resample(group, rng) for group in groups]) for _ in range(n_draws)]
        for metric in ("counts_r2", "support_auc", "support_leakage"):
            low, high = _percentile([item[metric] for item in draws])
            table.append({
                "dataset": dataset,
                "method": "FedAvg",
                "metric": metric,
                "estimate": point[metric],
                "low": low,
                "high": high,
            })
            print(f"  {dataset} {metric} {point[metric]:.3f} [{low:.3f}, {high:.3f}]", flush=True)
    return table


def fedavg_counts_ok(table):
    checks = []
    for row in table:
        if row["metric"] != "counts_r2":
            continue
        checks.append(row["estimate"] > 0.0 and row["low"] > 0.0)
    return bool(checks) and all(checks)


def _write_findings(table, passed):
    lines = ["", "## 2026-10-03", ""]
    lines.append(
        "Outcome for the FedPer body attacker. The score is the per-class move of the "
        "mean public feature from the body the client received to the body it uploaded. "
        "The line to the class count is fit on shadows. The observer is server-individual. "
        "The table is `results/tables/feature_shift.csv`."
    )
    lines.append("")
    for row in table:
        if row["metric"] != "counts_r2":
            continue
        lines.append(
            f"{row['dataset']} FedAvg counts R² {row['estimate']:.3f} "
            f"[{row['low']:.3f}, {row['high']:.3f}]."
        )
    if passed:
        lines.append("")
        lines.append(
            "The FedAvg control passed on every dataset, so FedPer can be read once those "
            "runs save the received body."
        )
    else:
        lines.append("")
        lines.append(
            "The FedAvg control did not pass on every dataset. FedPer stays unmeasured. "
            "This does not change the classifier result."
        )
    path = Path("findings.md")
    path.write_text(path.read_text(encoding="utf-8") + "\n".join(lines) + "\n", encoding="utf-8")


def main():
    import pandas as pd

    from fedseismic.config import resolve_device

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    grouped = {}
    for dataset, root in FEDAVG_ROOTS.items():
        for run in sorted(Path(root).glob("seed_*")):
            if not (run / "results.json").is_file():
                continue
            print(f"{dataset} {run.name}", flush=True)
            _clients, counts, shift, shadow_counts, shadow_shift = _collect(run, device)
            predicted = _predict(shadow_shift, shadow_counts, shift)
            rows = []
            for index in range(len(counts)):
                rows.append({
                    "counts": counts[index],
                    "pred": predicted[index],
                    "pi": to_simplex(counts[index]),
                })
            grouped.setdefault(dataset, []).append(rows)
    table = summarize(grouped)
    passed = fedavg_counts_ok(table)
    frame = pd.DataFrame(table)
    out = Path("results/tables/feature_shift.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    flag = Path("results/attacks/feature_shift_control.txt")
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("pass\n" if passed else "fail\n", encoding="utf-8")
    _write_findings(table, passed)
    print("FedAvg feature-shift control", "pass" if passed else "FAIL", flush=True)
    print(f"wrote {out.resolve()}")
    return passed


if __name__ == "__main__":
    main()
