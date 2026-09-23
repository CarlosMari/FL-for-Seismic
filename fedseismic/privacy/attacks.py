"""Property inference and membership scoring against a logged FL run."""

import numpy as np
from sklearn.metrics import roc_auc_score

from fedseismic.config import RunConfig, resolve_device

from .estimators import (
    HistogramRidge,
    clone_model_from_state,
    example_membership_scores,
    last_layer_prior,
    softmax_prior,
)
from .log import (
    latest_view_by_client,
    load_global_state,
    load_labels,
    load_local_states,
    load_meta,
    load_views,
)
from .metrics import (
    kl_divergence,
    leakage_from_tv,
    presence_f1,
    rare_presence_f1,
    to_simplex,
    total_variation,
)


def _unwrap_loader(entry):
    if entry is None:
        return None
    return entry[0] if isinstance(entry, tuple) else entry


def _summarize_pairs(pairs, rare_classes):
    if not pairs:
        return {
            "tv": float("nan"),
            "kl": float("nan"),
            "leakage": float("nan"),
            "presence_f1": float("nan"),
            "rare_presence_f1": float("nan"),
        }
    tvs = [total_variation(true, hat) for true, hat in pairs]
    return {
        "tv": float(np.mean(tvs)),
        "kl": float(np.mean([kl_divergence(true, hat) for true, hat in pairs])),
        "leakage": float(np.mean([leakage_from_tv(tv) for tv in tvs])),
        "presence_f1": float(np.mean([presence_f1(true, hat) for true, hat in pairs])),
        "rare_presence_f1": float(np.mean([
            rare_presence_f1(true, hat, rare_classes) for true, hat in pairs
        ])),
    }


def _mia_auc(member_scores, nonmember_scores):
    if len(member_scores) == 0 or len(nonmember_scores) == 0:
        return float("nan")
    labels = np.concatenate([
        np.ones(len(member_scores), dtype=np.int32),
        np.zeros(len(nonmember_scores), dtype=np.int32),
    ])
    scores = np.concatenate([member_scores, nonmember_scores])
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def evaluate_run(run_dir, data=None, device=None, max_probe_batches=None,
                 max_mia_examples=64):
    """Score distribution leakage and membership AUC for one seed directory."""
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    device = device or resolve_device(cfg.device)
    max_probe_batches = (
        cfg.privacy_probe_batches if max_probe_batches is None else max_probe_batches
    )
    histograms, rare_from_labels = load_labels(run_dir)
    rare_classes = tuple(cfg.rare_classes) or rare_from_labels
    views = load_views(run_dir)
    latest = latest_view_by_client(views)
    local_states = load_local_states(run_dir)
    global_state = load_global_state(run_dir)

    if data is None:
        from fedseismic.experiment import build_federated_run

        data = build_federated_run(cfg, meta.get("seed", cfg.seed))
    probe = _unwrap_loader(data.tests.get("test1"))
    factory = data.model_factory

    softmax_pairs = []
    last_layer_pairs = []
    global_pairs = []
    clients = sorted(set(histograms) & set(local_states))
    for client in clients:
        true = histograms[client]
        local_model = clone_model_from_state(factory, local_states[client], device=device)
        if probe is not None:
            softmax_pairs.append((true, softmax_prior(
                local_model, probe, cfg.num_classes, device, max_probe_batches,
            )))
        last_layer_pairs.append((true, last_layer_prior(local_states[client], cfg.num_classes)))
        if global_state is not None and probe is not None:
            global_model = clone_model_from_state(factory, global_state, device=device)
            global_pairs.append((true, softmax_prior(
                global_model, probe, cfg.num_classes, device, max_probe_batches,
            )))

    features, meta_hists, meta_clients = [], [], []
    for client, record in latest.items():
        if client not in histograms or "delta_proj" not in record:
            continue
        features.append(np.asarray(record["delta_proj"], dtype=np.float64).ravel())
        meta_hists.append(to_simplex(histograms[client]))
        meta_clients.append(client)
    if features:
        width = max(item.size for item in features)
        padded = np.stack([
            np.pad(item, (0, width - item.size)) if item.size < width else item[:width]
            for item in features
        ])
        predicted = HistogramRidge().predict_loo(padded, np.stack(meta_hists))
        meta_pairs = list(zip(meta_hists, predicted))
    else:
        meta_pairs = []

    mia_local = []
    mia_global = []
    global_model = (
        clone_model_from_state(factory, global_state, device=device)
        if global_state is not None else None
    )
    for client in clients:
        local_model = clone_model_from_state(factory, local_states[client], device=device)
        train_loader = data.loaders[client]
        # Same-distribution holdout. The public probe has a different class mix,
        # so using it as the non-member set scores label shift rather than membership.
        nonmember_loader = None
        client_tests = getattr(data, "client_tests", None)
        if client_tests and client < len(client_tests):
            holdout = client_tests[client]
            if holdout is not None and len(holdout.dataset):
                nonmember_loader = holdout
        member_loss, member_conf = example_membership_scores(
            local_model, train_loader, device, max_mia_examples,
        )
        if nonmember_loader is not None:
            non_loss, non_conf = example_membership_scores(
                local_model, nonmember_loader, device, max_mia_examples,
            )
            mia_local.append(_mia_auc(member_conf, non_conf))
            mia_local.append(_mia_auc(-member_loss, -non_loss))
        if global_model is not None:
            g_member_loss, g_member_conf = example_membership_scores(
                global_model, train_loader, device, max_mia_examples,
            )
            if nonmember_loader is not None:
                g_non_loss, g_non_conf = example_membership_scores(
                    global_model, nonmember_loader, device, max_mia_examples,
                )
                mia_global.append(_mia_auc(g_member_conf, g_non_conf))
                mia_global.append(_mia_auc(-g_member_loss, -g_non_loss))

    utility = meta.get("utility") or {}
    result = {
        "run_dir": str(run_dir),
        "algorithm": cfg.algorithm,
        "split": cfg.split,
        "partition_alpha": cfg.partition_alpha,
        "lambda_cap": cfg.lambda_cap,
        "fedkper_diversity": cfg.fedkper_diversity,
        "seed": meta.get("seed", cfg.seed),
        "num_clients_scored": len(clients),
        "softmax": _summarize_pairs(softmax_pairs, rare_classes),
        "last_layer": _summarize_pairs(last_layer_pairs, rare_classes),
        "meta_estimator": _summarize_pairs(meta_pairs, rare_classes),
        "global_softmax": _summarize_pairs(global_pairs, rare_classes),
        "mia_auc_local": float(np.nanmean(mia_local)) if mia_local else float("nan"),
        "mia_auc_global": float(np.nanmean(mia_global)) if mia_global else float("nan"),
        "local_mean": utility.get("local_mean"),
        "local_worst": utility.get("local_worst"),
        "global_on_local": utility.get("global_on_local"),
        "global_score": utility.get("global_score"),
    }
    local_mean = result["local_mean"]
    global_on_local = result["global_on_local"]
    if local_mean is not None and global_on_local is not None:
        result["personalization_gap"] = float(local_mean) - float(global_on_local)
    else:
        result["personalization_gap"] = float("nan")
    result["leakage"] = result["softmax"]["leakage"]
    return result


def flatten_attack_row(result):
    row = {
        "run_dir": result["run_dir"],
        "algorithm": result["algorithm"],
        "split": result["split"],
        "partition_alpha": result["partition_alpha"],
        "lambda_cap": result["lambda_cap"],
        "fedkper_diversity": result["fedkper_diversity"],
        "seed": result["seed"],
        "leakage": result["leakage"],
        "personalization_gap": result["personalization_gap"],
        "local_mean": result["local_mean"],
        "local_worst": result["local_worst"],
        "global_on_local": result["global_on_local"],
        "global_score": result["global_score"],
        "mia_auc_local": result["mia_auc_local"],
        "mia_auc_global": result["mia_auc_global"],
    }
    for name in ("softmax", "last_layer", "meta_estimator", "global_softmax"):
        for key, value in result[name].items():
            row[f"{name}_{key}"] = value
    return row
