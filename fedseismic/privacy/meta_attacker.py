"""Learned attacker on the bias change and the row size.

Shadow clients are a fresh Dirichlet split. Each one trains with the victim's
loss for the same number of local epochs, starting from that seed's final
global model. The attacker is fit only on those shadows. Real clients are
scored from the classifier delta they actually uploaded against the model
they received.

The mix attacker is a ridge from the two clues to the class fractions. The
presence attacker is a balanced logistic regression per class. A third score
asks whether a client is in the absent group or a high-share group. That is
the distinguishing game. None of the fits sees a real client.
"""

from pathlib import Path

import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from fedseismic.config import RunConfig, resolve_device
from fedseismic.data import (
    build_cifar_loaders, build_medmnist_loaders, load_cifar10, load_medmnist, partition_dirichlet,
)
from fedseismic.data.partition import drop_indices, holdout_balanced_pad
from fedseismic.federated.client import ClientTrainer
from fedseismic.privacy.head_channels import RUNS, SEEDS, _client_id
from fedseismic.privacy.log import load_global_state, load_labels, load_meta
from fedseismic.privacy.metrics import to_simplex
from fedseismic.privacy.row_delta import _summarize
from fedseismic.privacy.weight_attack import layer_delta

ABSENT = 1e-3
HIGH_SHARE = 0.25
# Midpoints between 0.01, 0.05, and 0.10. The last bin is the named threshold.
# A share in (0.175, 0.25) is in neither bin.
SHARE_BINS = (
    ("0.01", 1e-3, 0.03),
    ("0.05", 0.03, 0.075),
    ("0.10", 0.075, 0.175),
    (">=0.25", 0.25, None),
)


def tpr_at_fpr(labels, scores, fpr_cap=0.1):
    """Highest true-positive rate among thresholds whose false-positive rate is at most ``fpr_cap``."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    allowed = tpr[fpr <= fpr_cap + 1e-12]
    if len(allowed) == 0:
        return 0.0
    return float(allowed.max())


def participation_features(run_dir):
    """One ``[row sizes, bias change]`` vector per participation, grouped by client."""
    grouped = {}
    for path in sorted((Path(run_dir) / "head").glob("round_*_client_*.npz")):
        with np.load(path) as handle:
            weight = np.asarray(handle["weight_delta"], dtype=np.float64)
            bias = np.asarray(handle["bias_delta"], dtype=np.float64).reshape(-1)
        sizes = np.linalg.norm(weight.reshape(weight.shape[0], -1), axis=1)
        grouped.setdefault(_client_id(path), []).append(np.concatenate([sizes, bias]))
    if not grouped:
        raise FileNotFoundError(f"no head deltas in {run_dir}")
    return grouped


def training_pool(cfg):
    """Images and labels the shadow clients are drawn from, plus the channel count."""
    if str(cfg.dataset) == "cifar10":
        images, targets, _, _ = load_cifar10(cfg.data_root, download=False)
        return images, targets, 3
    images, targets, _, _, spec = load_medmnist(cfg.dataset, cfg.data_root, download=False)
    return images, targets, int(spec["n_channels"])


def _shadow_loaders(cfg, images, targets, n_channels, n_shadows, seed, cohort_size=None):
    """Shadow client loaders at victim scale.

    ``cohort_size`` partitions the pool the way the victims were split (N
    clients per draw, repeated until ``n_shadows`` loaders exist) instead of
    one ``n_shadows``-way split. Without it a 48-way shadow split makes each
    shadow ~pool/48 while an N=2 victim holds ~pool/2, and the count line is
    calibrated at the wrong scale. The default (None) keeps the old behavior
    for the 20/30-client runs, where the mismatch is mild.
    """
    per_class = int(getattr(cfg, "public_pad_per_class", 0) or 0)
    pool = targets
    if per_class > 0:
        pad = holdout_balanced_pad(targets, per_class, np.random.RandomState(int(seed) + 17001))
        pool, kept = drop_indices(targets, pad)
    else:
        kept = None
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
        parts = [[int(kept[index]) for index in part] for part in parts]
    parts = [part for part in parts if len(part) >= 2][: int(n_shadows)]
    if str(cfg.dataset) == "cifar10":
        return build_cifar_loaders(images, targets, parts, cfg.batch_size, train=True, num_workers=2)
    return build_medmnist_loaders(
        images, targets, parts, cfg.batch_size, n_channels, train=True, num_workers=2,
    )


def _mix(loader, num_classes):
    labels = np.asarray(loader.dataset.targets)[list(loader.dataset.indices)]
    return to_simplex(np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64))


def make_shadow_trainer(cfg, model, loader, device, public_loader=None):
    """The victim's local trainer. FedKPer shadows distill; the others do not."""
    kwargs = dict(
        local_epochs=cfg.local_epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        optimizer=cfg.optimizer,
        momentum=cfg.momentum,
        num_classes=cfg.num_classes,
        logit_adjust=bool(cfg.logit_adjust),
        prior_smooth=float(getattr(cfg, "prior_smooth", 0.0)),
        class_reweight=bool(getattr(cfg, "class_reweight", False)),
        public_loader=public_loader if getattr(cfg, "train_public_pad", False) else None,
    )
    if getattr(cfg, "local_steps", None):
        from fedseismic.federated.client import StepClientTrainer
        return StepClientTrainer(
            model, loader, nn.CrossEntropyLoss(), device=device,
            local_steps=int(cfg.local_steps),
            gradient_balance=str(getattr(cfg, "gradient_balance", "example")),
            **kwargs,
        )
    if str(getattr(cfg, "gradient_balance", "example")) == "class_minibatch":
        from fedseismic.federated.client import MinibatchBalancedTrainer
        return MinibatchBalancedTrainer(
            model, loader, nn.CrossEntropyLoss(), device=device, **kwargs,
        )
    if str(getattr(cfg, "algorithm", "")) == "fedkper":
        from fedseismic.federated.fedkper import FedKPerClientTrainer
        return FedKPerClientTrainer(
            model, loader, nn.CrossEntropyLoss(), device=device,
            lambda_cap=float(cfg.lambda_cap), grad_clip=float(cfg.grad_clip), **kwargs,
        )
    return ClientTrainer(model, loader, nn.CrossEntropyLoss(), device=device, **kwargs)


def collect_shadows(cfg, global_state, model_factory, images, targets, n_channels, public_loader, device, n_shadows, seed, start_states=None, return_positive=False, cohort_size=None, start_round_max=None):
    """Shadow updates with the victim's trainer.

    ``start_states`` are the received models, one per round. ``start_round_max``
    restricts starts to rounds 1..that number: late-round starts come from a
    converged model and carry almost no count slope, so a uniform draw over 40
    rounds dilutes the fit with converged noise. Victim means are dominated by
    early rounds by magnitude, and shadows must live where the signal is.
    """
    features = []
    mixes = []
    positives = []
    starts = list(start_states) if start_states else None
    if starts is not None and start_round_max is not None:
        starts = starts[: max(int(start_round_max), 1)]
    pick = np.random.RandomState(40_000 + int(seed))
    for loader in _shadow_loaders(cfg, images, targets, n_channels, n_shadows, seed, cohort_size=cohort_size):
        state = global_state if not starts else starts[int(pick.randint(len(starts)))]
        trainer = make_shadow_trainer(cfg, model_factory(), loader, device, public_loader)
        trainer.download(state)
        import torch
        torch.manual_seed(50_000 + int(seed) + len(features))
        trainer.train(global_state=state)
        uploaded = trainer.upload()
        feature = layer_delta(uploaded, state, cfg.num_classes)
        if feature is None:
            continue
        features.append(feature)
        mixes.append(_mix(loader, cfg.num_classes))
        if return_positive:
            from fedseismic.privacy.decaf import positive_sum
            from fedseismic.privacy.estimators import extract_last_linear
            local_weight, _ = extract_last_linear(uploaded)
            received_weight, _ = extract_last_linear(state)
            delta = np.asarray(local_weight, dtype=np.float64) - np.asarray(received_weight, dtype=np.float64)
            positives.append(positive_sum(delta)[:cfg.num_classes])
    if return_positive:
        return np.stack(features), np.stack(mixes), np.stack(positives)
    return np.stack(features), np.stack(mixes)


def _predict_client(model, scaler, rows):
    transformed = scaler.transform(np.stack(rows))
    return model.predict(transformed).mean(axis=0)


def _presence_model(features, mixes):
    """Per-class probability that the class is present. Fit on shadows only."""
    present = mixes > ABSENT
    models = []
    for class_index in range(mixes.shape[1]):
        labels = present[:, class_index].astype(int)
        if len(np.unique(labels)) < 2:
            models.append(None)
            continue
        classifier = LogisticRegression(class_weight="balanced", max_iter=500)
        classifier.fit(features, labels)
        models.append(classifier)
    return models


def _presence_scores(models, scaler, grouped, clients):
    scores = []
    for client in clients:
        transformed = scaler.transform(np.stack(grouped[client]))
        row = np.zeros(len(models), dtype=np.float64)
        for class_index, classifier in enumerate(models):
            if classifier is None:
                row[class_index] = 0.5
            else:
                row[class_index] = float(classifier.predict_proba(transformed)[:, 1].mean())
        scores.append(row)
    return scores


def in_share_bin(share, lo, hi):
    """True when ``share`` falls in ``(lo, hi]``, or in ``[lo, 1]`` when ``hi`` is None."""
    share = float(share)
    if hi is None:
        return share >= float(lo)
    return share > float(lo) and share <= float(hi)


def _game_masks(mixes, class_index, negative, positive):
    column = mixes[:, class_index]
    low = np.array([negative(value) for value in column])
    high = np.array([positive(value) for value in column])
    return low, high


def _fit_game(features, mixes, real_features, real_mixes, negative, positive, min_shadow=4, min_real=2):
    """Accuracy and AUC of a shadow-fit classifier, averaged over classes with enough rows."""
    accuracies = []
    aucs = []
    n_real_low = []
    n_real_high = []
    for class_index in range(mixes.shape[1]):
        shadow_low, shadow_high = _game_masks(mixes, class_index, negative, positive)
        real_low, real_high = _game_masks(real_mixes, class_index, negative, positive)
        if int(shadow_low.sum()) < min_shadow or int(shadow_high.sum()) < min_shadow:
            continue
        if int(real_low.sum()) < min_real or int(real_high.sum()) < min_real:
            continue
        train_x = np.concatenate([features[shadow_low], features[shadow_high]])
        train_y = np.concatenate([
            np.zeros(int(shadow_low.sum())), np.ones(int(shadow_high.sum())),
        ])
        classifier = LogisticRegression(class_weight="balanced", max_iter=500)
        classifier.fit(train_x, train_y)
        test_x = np.concatenate([real_features[real_low], real_features[real_high]])
        test_y = np.concatenate([
            np.zeros(int(real_low.sum())), np.ones(int(real_high.sum())),
        ])
        accuracies.append(float(classifier.score(test_x, test_y)))
        if len(np.unique(test_y)) < 2:
            continue
        aucs.append(float(roc_auc_score(test_y, classifier.predict_proba(test_x)[:, 1])))
        n_real_low.append(int(real_low.sum()))
        n_real_high.append(int(real_high.sum()))
    return {
        "accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
        "auc": float(np.mean(aucs)) if aucs else float("nan"),
        "n_classes": int(len(accuracies)),
        "n_real_absent": int(np.sum(n_real_low)) if n_real_low else 0,
        "n_real_bin": int(np.sum(n_real_high)) if n_real_high else 0,
    }


def share_accuracy(features, mixes, real_features, real_mixes, low=ABSENT, high=HIGH_SHARE):
    """Fraction of real clients placed on the correct side of absent versus high share."""
    scored = _fit_game(
        features, mixes, real_features, real_mixes,
        negative=lambda share: share <= low,
        positive=lambda share: share >= high,
    )
    return scored["accuracy"]


def share_bin_metrics(features, mixes, real_features, real_mixes, lo, hi, absent=ABSENT):
    """Learned attacker on absent classes versus shares inside one bin."""
    return _fit_game(
        features, mixes, real_features, real_mixes,
        negative=lambda share: share <= absent,
        positive=lambda share: in_share_bin(share, lo, hi),
    )


def _class_clues(features, mixes, lo, hi, absent=ABSENT):
    """One row per class that is absent or inside the bin: that class's row size and bias."""
    features = np.asarray(features, dtype=np.float64)
    mixes = np.asarray(mixes, dtype=np.float64)
    num_classes = mixes.shape[1]
    clues = []
    labels = []
    for row, mix in zip(features, mixes):
        for class_index, share in enumerate(mix):
            if share <= absent:
                labels.append(0)
            elif in_share_bin(share, lo, hi):
                labels.append(1)
            else:
                continue
            clues.append([row[class_index], row[num_classes + class_index]])
    if not clues:
        return np.zeros((0, 2)), np.zeros(0)
    return np.asarray(clues, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def pooled_bin_metrics(features, mixes, real_features, real_mixes, lo, hi, absent=ABSENT):
    """One classifier for every class, using only that class's row size and bias.

    The per-class attacker skips a bin when a single class rarely lands in it.
    This pools those rows. It is still fit only on shadows.
    """
    train_x, train_y = _class_clues(features, mixes, lo, hi, absent)
    test_x, test_y = _class_clues(real_features, real_mixes, lo, hi, absent)
    if len(train_y) < 8 or len(np.unique(train_y)) < 2 or len(test_y) < 4 or len(np.unique(test_y)) < 2:
        return {"accuracy": float("nan"), "auc": float("nan"), "n_real_absent": 0, "n_real_bin": 0}
    classifier = LogisticRegression(class_weight="balanced", max_iter=500)
    classifier.fit(train_x, train_y)
    return {
        "accuracy": float(classifier.score(test_x, test_y)),
        "auc": float(roc_auc_score(test_y, classifier.predict_proba(test_x)[:, 1])),
        "n_real_absent": int((test_y == 0).sum()),
        "n_real_bin": int((test_y == 1).sum()),
    }


def _pooled_tpr(truths, scores):
    labels = []
    values = []
    for truth, score in zip(truths, scores):
        labels.extend((to_simplex(truth) > ABSENT).astype(int).tolist())
        values.extend(np.asarray(score, dtype=np.float64).tolist())
    return tpr_at_fpr(labels, values, fpr_cap=0.1)


def score_seed(run_dir, device, n_shadows):
    from fedseismic.experiment import build_federated_run

    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    seed = int(meta.get("seed", cfg.seed))
    data = build_federated_run(cfg, seed)
    images, targets, _, _, spec = load_medmnist(cfg.dataset, cfg.data_root, download=False)
    global_state = load_global_state(run_dir)
    histograms, _ = load_labels(run_dir)
    grouped = participation_features(run_dir)
    clients = sorted(client for client in grouped if client in histograms)
    truths = [to_simplex(histograms[client]) for client in clients]

    public = data.public_loader if cfg.train_public_pad else None
    shadow_x, shadow_y = collect_shadows(
        cfg, global_state, data.model_factory, images, targets, spec["n_channels"],
        public, device, n_shadows, seed,
    )
    scaler = StandardScaler().fit(shadow_x)
    scaled = scaler.transform(shadow_x)
    mix_model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(scaled, shadow_y)
    mix_hat = [_predict_client(mix_model, scaler, grouped[client]) for client in clients]
    presence_models = _presence_model(scaled, shadow_y)
    presence_hat = _presence_scores(presence_models, scaler, grouped, clients)
    real_mean = scaler.transform(np.stack([np.mean(grouped[client], axis=0) for client in clients]))
    return {
        "n_shadows": int(len(shadow_x)),
        "learned_mix": _summarize(truths, mix_hat),
        "learned_presence": {
            **_summarize(truths, presence_hat),
            "tpr_at_10": _pooled_tpr(truths, presence_hat),
            "share_acc": share_accuracy(scaled, shadow_y, real_mean, np.stack(truths)),
        },
        "hand_bias": _summarize(truths, [np.mean(grouped[client], axis=0)[cfg.num_classes:] for client in clients]),
        "hand_row": _summarize(truths, [np.mean(grouped[client], axis=0)[:cfg.num_classes] for client in clients]),
    }


def main():
    import pandas as pd

    device = resolve_device("cuda")
    rows = []
    for (dataset, method), relative in RUNS.items():
        for seed in SEEDS:
            print(f"{dataset} {method} {seed}", flush=True)
            scored = score_seed(Path(relative) / seed, device, n_shadows=48)
            n_shadows = scored.pop("n_shadows")
            for attack, summary in scored.items():
                row = {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "attack": attack,
                    "n_shadows": n_shadows,
                    "pearson": summary["pearson"],
                    "presence_auc": summary["presence_auc"],
                    "loo_r2": summary["loo_r2"],
                    "r2": summary["r2"],
                    "tpr_at_10": summary.get("tpr_at_10", float("nan")),
                    "share_acc": summary.get("share_acc", float("nan")),
                }
                rows.append(row)
                extra = ""
                if attack == "learned_presence":
                    extra = f" tpr10 {summary['tpr_at_10']:.3f} share {summary['share_acc']:.3f}"
                print(
                    f"  {attack} pearson {summary['pearson']:+.3f} "
                    f"auc {summary['presence_auc']:.3f} loo {summary['loo_r2']:+.3f}{extra}",
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    output = Path("results/tables/meta_attacker.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    grouped = frame.groupby(["method", "attack"])[
        ["pearson", "presence_auc", "loo_r2", "tpr_at_10", "share_acc"]
    ].mean()
    print(grouped.to_string())
    print(f"wrote {output.resolve()}")


if __name__ == "__main__":
    main()
