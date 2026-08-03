import numpy as np

from fedseismic.data.partition import partition_iid, partition_noniid


def test_partition_equivalence_gate_for_requested_client_counts():
    for count in (3, 5, 20):
        assert partition_noniid(20, count) == [
            list(range(i * (20 // count), (i + 1) * (20 // count)))
            if i < count - 1 else list(range(i * (20 // count), 20))
            for i in range(count)
        ]
        rng = np.random.RandomState(123)
        actual = partition_iid(20, count, rng)
        expected_rng = np.random.RandomState(123)
        values = np.arange(20)
        expected_rng.shuffle(values)
        size = 20 // count
        expected = [sorted(values[i * size:(i + 1) * size].tolist())
                    if i < count - 1 else sorted(values[i * size:].tolist())
                    for i in range(count)]
        assert actual == expected
