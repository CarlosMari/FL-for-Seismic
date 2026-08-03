"""Aggregation strategies used by the segmentation server."""

from copy import deepcopy

import numpy as np

RARE_CLASSES = (4, 5)
TRAIN_CLASS_PRIOR = np.array(
    [0.28093788, 0.1188557, 0.48592013, 0.0664164, 0.03278635, 0.01508355],
    dtype=np.float64,
)
RARE_FRAC_THRESHOLD = 1e-3


def aggregate_state_dicts(state_dicts, weights=None):
    if not state_dicts:
        raise ValueError("cannot aggregate an empty collection")
    if weights is None:
        averaged = deepcopy(state_dicts[0])
        for key in averaged:
            for index in range(1, len(state_dicts)):
                averaged[key] = averaged[key] + state_dicts[index][key]
            averaged[key] = averaged[key] / len(state_dicts)
        return averaged
    w = np.asarray(weights, dtype=np.float64)
    if len(w) != len(state_dicts) or not np.isfinite(w).all() or w.sum() == 0:
        raise ValueError("aggregation weights must be finite and sum to a non-zero value")
    w = w / w.sum()
    averaged = deepcopy(state_dicts[0])
    for key in averaged:
        averaged[key] = averaged[key] * w[0]
        for index in range(1, len(state_dicts)):
            averaged[key] = averaged[key] + state_dicts[index][key] * w[index]
    return averaged


def is_bn_key(key):
    parts = key.split(".")
    return any(
        part == "double_conv" and index + 1 < len(parts) and parts[index + 1] in ("1", "4")
        for index, part in enumerate(parts)
    )


class EqualAgg:
    def weights(self, selected_clients, **kwargs):
        return [1.0] * len(selected_clients)

    def aggregate(self, state_dicts, **kwargs):
        return aggregate_state_dicts(state_dicts)


class DiversityAgg:
    def weights(self, selected_clients, client_info, **kwargs):
        return [client_info[c]["num_classes"] for c in selected_clients]


class RareAgg:
    def weights(self, selected_clients, client_info, **kwargs):
        return [max(client_info[c]["rare_fraction"], 0.01) for c in selected_clients]


class RareMiouAgg:
    def weights(self, selected_clients, client_info, client_class_ious=None, **kwargs):
        if client_class_ious is None:
            raise AssertionError("rare_miou needs client_class_ious")
        weights = []
        for index, client in enumerate(selected_clients):
            fracs = client_info[client]["class_fracs"]
            gated = [client_class_ious[index][c] for c in RARE_CLASSES
                     if fracs[c] > RARE_FRAC_THRESHOLD]
            miou = float(np.mean(gated)) if gated else 0.0
            weights.append((sum(fracs[c] for c in RARE_CLASSES) + 1e-3) * (miou + 1e-3))
        return weights


class InvFreqMiouAgg:
    def weights(self, selected_clients, client_info, client_class_ious=None, **kwargs):
        if client_class_ious is None:
            raise AssertionError("invfreq_miou needs client_class_ious")
        result = []
        for index, client in enumerate(selected_clients):
            fracs = client_info[client]["class_fracs"]
            ious = client_class_ious[index]
            value = sum(
                (fracs[c] / max(TRAIN_CLASS_PRIOR[c], 1e-8)) * ious[c]
                for c in RARE_CLASSES if fracs[c] > RARE_FRAC_THRESHOLD
            )
            result.append(value + 1e-3)
        return result


class InvFreqAgg:
    def weights(self, selected_clients, client_info, **kwargs):
        return [
            sum(client_info[c]["class_fracs"][rc] / max(TRAIN_CLASS_PRIOR[rc], 1e-8)
                for rc in RARE_CLASSES) + 1e-3
            for c in selected_clients
        ]


class InvFreqInvMiouAgg:
    def weights(self, selected_clients, client_info, client_class_ious=None, **kwargs):
        if client_class_ious is None:
            raise AssertionError("invfreq_invmiou needs client_class_ious")
        result = []
        for index, client in enumerate(selected_clients):
            fracs = client_info[client]["class_fracs"]
            ious = client_class_ious[index]
            value = sum(
                (fracs[c] / max(TRAIN_CLASS_PRIOR[c], 1e-8)) * (1.0 - ious[c])
                for c in RARE_CLASSES if fracs[c] > RARE_FRAC_THRESHOLD
            )
            result.append(value + 1e-3)
        return result


class AccuracyAgg:
    def weights(self, selected_clients, client_models=None, test_loader=None,
                test_labels=None, device=None, **kwargs):
        if client_models is None or test_loader is None:
            raise AssertionError("accuracy needs client_models and test_loader")
        from fedseismic.eval.metrics import evaluate
        return [max(evaluate(model, test_loader, test_labels, device)[0], 0.01)
                for model in client_models]


AGGREGATORS = {
    "equal": EqualAgg,
    "diversity": DiversityAgg,
    "rare": RareAgg,
    "rare_miou": RareMiouAgg,
    "invfreq_miou": InvFreqMiouAgg,
    "invfreq": InvFreqAgg,
    "invfreq_invmiou": InvFreqInvMiouAgg,
    "accuracy": AccuracyAgg,
}


def get_agg_weights(strategy, selected_clients, client_info, **kwargs):
    try:
        return AGGREGATORS[strategy]().weights(
            selected_clients=selected_clients, client_info=client_info, **kwargs
        )
    except KeyError as exc:
        raise ValueError(f"unknown aggregation strategy: {strategy}") from exc
