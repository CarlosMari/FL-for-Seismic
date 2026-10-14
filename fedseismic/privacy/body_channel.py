"""Whether the uploaded body reveals the client's class mix.

``local/client_*.pt`` is the state ``log_upload`` stored after local training.
The probe fits a class-balanced logistic regression on half of the public
probe and reads per-class recall on the other half. Where the run saved the
model the client received, the score is that recall minus the same recall on
the received body. Otherwise each class's recall has the mean across clients
removed. FedAvg is scored first.
"""

from pathlib import Path
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from fedseismic.config import RunConfig, resolve_device
from fedseismic.privacy.disk_checks import _class_scores
from fedseismic.privacy.estimators import clone_model_from_state
from fedseismic.privacy.log import _load_checkpoint, load_labels, load_local_states, load_meta
from fedseismic.privacy.mechanism import prior_presence_auc, bootstrap_seed_mean
from fedseismic.privacy.metrics import to_simplex
from fedseismic.privacy.row_delta import _feature_table, _loo_r2, _pearson, _presence_auc

ROOTS = {
    ("Blood", "FedAvg"): "results/runs/prior_smooth_blood/fedavg_noniid_a0.1_lNA_dna_p1",
    ("OrganC", "FedAvg"): "results/runs/prior_smooth_organc/fedavg_noniid_a0.1_lNA_dna_p1",
    ("CIFAR-10", "FedAvg"): "results/runs/prior_smooth_cifar/fedavg_noniid_a0.1_lNA_dna_p1",
    ("Blood", "FedPer"): "results/runs/privacy_fedper/fedper_noniid_a0.1_lNA_dna",
    ("OrganC", "FedPer"): "results/runs/privacy_fedper_organc/fedper_noniid_a0.1_lNA_dna",
    ("CIFAR-10", "FedPer"): "results/runs/privacy_cifar/fedper_noniid_a0.1_lNA_dna",
    ("Blood", "FedBABU"): "results/runs/fedbabu_blood/fedbabu_noniid_a0.1_lNA_dna",
    ("OrganC", "FedBABU"): "results/runs/fedbabu_organc/fedbabu_noniid_a0.1_lNA_dna",
    ("CIFAR-10", "FedBABU"): "results/runs/fedbabu_cifar/fedbabu_noniid_a0.1_lNA_dna",
}


def center_rows(scores):
    """Subtract the per-class mean. The server sees every client's upload."""
    stacked = np.vstack(scores)
    return stacked - stacked.mean(axis=0)


def last_rounds(run_dir):
    """Last participation round of each client, matching ``received/round_XXX.pt``."""
    last = {}
    path = Path(run_dir) / "round_scores.jsonl"
    if not path.is_file():
        return last
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            client = int(record["client"])
            rnd = int(record["round"])
            last[client] = max(last.get(client, 0), rnd)
    return last


def _recall(model, loader, device, num_classes, split):
    features, labels = _feature_table(model, loader, device)
    fit_index, score_index = split
    scaler = StandardScaler()
    classifier = LogisticRegression(class_weight="balanced", max_iter=500, solver="lbfgs")
    classifier.fit(scaler.fit_transform(features[fit_index]), labels[fit_index])
    predicted = np.asarray(classifier.predict(scaler.transform(features[score_index])))
    decision = np.asarray(classifier.decision_function(scaler.transform(features[score_index])), dtype=np.float64)
    if decision.ndim == 1:
        decision = np.column_stack([-decision, decision])
    recall, _margin = _class_scores(decision, predicted, labels[score_index], num_classes)
    return recall, labels


def _split(labels):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    fit_index, score_index = next(splitter.split(np.zeros(len(labels)), labels))
    return fit_index, score_index


def probe_run(run_dir, device):
    """Per-client scores. Difference when the received model exists, otherwise centered recall."""
    from fedseismic.experiment import build_federated_run

    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    data = build_federated_run(cfg, meta.get("seed", cfg.seed))
    histograms, _ = load_labels(run_dir)
    local_states = load_local_states(run_dir)
    rounds = last_rounds(run_dir)
    clients = sorted(client for client in local_states if client in histograms)
    use_difference = any((run_dir / "received").glob("round_*.pt"))
    received_cache = {}
    split = None
    rows = []
    for client in clients:
        received_path = run_dir / "received" / f"round_{rounds.get(client, 0):03d}.pt"
        if use_difference and not received_path.is_file():
            continue
        model = clone_model_from_state(data.model_factory, local_states[client], device=device)
        if split is None:
            _features, labels = _feature_table(model, data.probe_loader, device)
            split = _split(labels)
        recall, _ = _recall(model, data.probe_loader, device, cfg.num_classes, split)
        print(f"  client {client}", flush=True)
        if not np.isfinite(recall).all():
            raise ValueError(f"probe recall missing a class for client {client}")
        if use_difference:
            if received_path not in received_cache:
                received_model = clone_model_from_state(
                    data.model_factory, _load_checkpoint(received_path), device=device,
                )
                received_cache[received_path] = _recall(
                    received_model, data.probe_loader, device, cfg.num_classes, split,
                )[0]
            score = recall - received_cache[received_path]
        else:
            score = recall
        rows.append({
            "client": client,
            "pi": to_simplex(histograms[client]),
            "score": np.asarray(score, dtype=np.float64),
        })
    if not rows:
        raise FileNotFoundError(f"no client bodies in {run_dir}")
    mode = "difference" if use_difference else "center"
    if mode == "center":
        centered = center_rows([row["score"] for row in rows])
        for row, score in zip(rows, centered):
            row["score"] = score
    return rows, mode


def _presence(rows):
    return _presence_auc([row["pi"] for row in rows], [row["score"] for row in rows])


def _pearson_rows(rows):
    return _pearson([row["pi"] for row in rows], [row["score"] for row in rows])


def _loo(rows):
    return _loo_r2([row["pi"] for row in rows], [row["score"] for row in rows])


def summarize_runs(grouped, n_draws=1000):
    """``grouped`` maps (dataset, method) to a list of per-seed row lists."""
    table = []
    for (dataset, method), groups in grouped.items():
        mode = groups[0]["mode"]
        seeds = [item["rows"] for item in groups]
        auc, low, high, _ = bootstrap_seed_mean(seeds, _presence, n_draws=n_draws)
        floor, _, _, _ = bootstrap_seed_mean(seeds, prior_presence_auc, n_draws=n_draws)
        pearson, pearson_low, pearson_high, _ = bootstrap_seed_mean(seeds, _pearson_rows, n_draws=n_draws)
        loo, loo_low, loo_high, _ = bootstrap_seed_mean(seeds, _loo, n_draws=n_draws)
        table.append({
            "dataset": dataset,
            "method": method,
            "score": mode,
            "presence_auc": auc,
            "low": low,
            "high": high,
            "floor": floor,
            "leakage": auc - floor,
            "pearson": pearson,
            "pearson_low": pearson_low,
            "pearson_high": pearson_high,
            "loo_r2": loo,
            "loo_r2_low": loo_low,
            "loo_r2_high": loo_high,
        })
    return table


def fedavg_ok(table):
    checks = []
    for row in table:
        if row["method"] != "FedAvg":
            continue
        checks.append(row["presence_auc"] > row["floor"] and row["low"] > row["floor"])
    return bool(checks) and all(checks)


def _saved_path(run):
    relative = Path(run).resolve().relative_to(Path("results/runs").resolve())
    return Path("results/attacks/body") / relative / "probe.npz"


def _load_probe(path):
    with np.load(path) as handle:
        clients = np.asarray(handle["client"])
        pi = np.asarray(handle["pi"])
        score = np.asarray(handle["score"])
        mode = handle["mode"].item()
    if isinstance(mode, bytes):
        mode = mode.decode()
    rows = [
        {"client": int(clients[index]), "pi": pi[index], "score": score[index]}
        for index in range(len(clients))
    ]
    return rows, str(mode)


def _seed_dirs(root):
    root = Path(root)
    found = sorted(path for path in root.glob("seed_*") if (path / "results.json").is_file())
    if not found:
        raise FileNotFoundError(f"no seeds in {root}")
    return found


def main():
    import pandas as pd

    device = resolve_device("cuda")
    order = [("Blood", "FedAvg"), ("OrganC", "FedAvg"), ("CIFAR-10", "FedAvg")]
    others = [key for key in ROOTS if key not in order]
    grouped = {}

    def consume(keys):
        for key in keys:
            dataset, method = key
            for run in _seed_dirs(ROOTS[key]):
                print(f"{dataset} {method} {run.name}", flush=True)
                saved = _saved_path(run)
                if saved.is_file():
                    client_rows, mode = _load_probe(saved)
                    print(f"  loaded {mode} clients {len(client_rows)}", flush=True)
                else:
                    client_rows, mode = probe_run(run, device)
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        saved,
                        client=np.asarray([row["client"] for row in client_rows], dtype=np.int32),
                        pi=np.stack([row["pi"] for row in client_rows]),
                        score=np.stack([row["score"] for row in client_rows]),
                        mode=np.asarray(mode),
                    )
                    print(f"  {mode} clients {len(client_rows)}", flush=True)
                grouped.setdefault(key, []).append({"rows": client_rows, "mode": mode})

    consume(order)
    fedavg_table = summarize_runs({key: grouped[key] for key in order})
    ok = fedavg_ok(fedavg_table)
    print("FedAvg body control", "pass" if ok else "FAIL", flush=True)
    for row in fedavg_table:
        print(
            f"  {row['dataset']} auc {row['presence_auc']:.3f} "
            f"[{row['low']:.3f}, {row['high']:.3f}] floor {row['floor']:.3f}",
            flush=True,
        )
    if ok:
        consume(others)
    table = summarize_runs(grouped) if ok else fedavg_table
    frame = pd.DataFrame(table)
    path = Path("results/tables/body_probe.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(frame.to_string(index=False))
    print(f"wrote {path.resolve()}")
    if not ok:
        print("STOP: the body probe did not read FedAvg. FedPer and FedBABU are not interpreted.")


if __name__ == "__main__":
    main()
