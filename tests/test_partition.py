import numpy as np

from fedseismic.data.partition import compute_client_class_info, partition_iid, partition_noniid


def test_geographic_partitions_match_legacy_shape():
    for clients in (3, 5, 20):
        partitions = partition_noniid(20, clients)
        assert sorted(index for part in partitions for index in part) == list(range(20))
        assert partitions[-1][-1] == 19


def test_iid_partition_is_sorted_with_seeded_random_state():
    first = partition_iid(20, 5, np.random.RandomState(42))
    second = partition_iid(20, 5, np.random.RandomState(42))
    assert first == second
    assert all(part == sorted(part) for part in first)


def test_client_class_fractions():
    labels = np.array([
        [[0, 1], [4, 5]],
        [[0, 0], [4, 5]],
    ])
    info = compute_client_class_info(labels, [[0], [1]])
    assert info[0]["class_fracs"].tolist() == [0.75, 0.25, 0.0, 0.0, 0.0, 0.0]
    assert info[1]["rare_fraction"] == 1.0
