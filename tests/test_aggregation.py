import numpy as np
import torch

from fedseismic.federated.aggregation import (
    AGGREGATORS,
    aggregate_state_dicts,
    get_agg_weights,
)


def test_equal_aggregation_is_arithmetic_mean():
    states = [{"weight": torch.tensor([1.0, 3.0])}, {"weight": torch.tensor([3.0, 5.0])}]
    assert torch.equal(aggregate_state_dicts(states)["weight"], torch.tensor([2.0, 4.0]))


def test_all_eight_strategies_are_registered():
    assert len(AGGREGATORS) == 8
    info = [
        {"num_classes": 2, "rare_fraction": 0.0, "class_fracs": np.zeros(6)},
        {"num_classes": 6, "rare_fraction": 0.2, "class_fracs": np.array([0, 0, 0, 0, .1, .1])},
    ]
    ious = [np.zeros(6), np.ones(6)]
    for strategy in AGGREGATORS:
        kwargs = {"client_class_ious": ious} if "miou" in strategy else {}
        if strategy == "accuracy":
            continue
        weights = get_agg_weights(strategy, [0, 1], info, **kwargs)
        assert len(weights) == 2
        assert np.isfinite(weights).all()
