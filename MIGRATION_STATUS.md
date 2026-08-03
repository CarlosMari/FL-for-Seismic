# Legacy migration status

Evidence is based on the committed equivalence tests and their executed gate
results. A script is safe to delete only when its complete round-1 state gate
passes.

| legacy script | fedseismic equivalent | round-1 gate | safe to delete |
|---|---|---|---|
| `train_federated.py` | `Server` + `ClientTrainer` | PASS: `tests/test_round_equivalence.py` | YES |
| `train_fedprox.py` | `Server` + `FedProxClientTrainer` | PASS: `tests/test_variant_round_equivalence.py::test_fedprox_round_one_is_bit_exact` | YES |
| `train_fedbn.py` | `Server` + `FedBNClientTrainer` | PASS: `tests/test_variant_round_equivalence.py::test_fedbn_round_one_is_bit_exact` | YES |
| `train_fedvls.py` | `Server` + `FedVLSClientTrainer` | PASS: `tests/test_variant_round_equivalence.py::test_fedvls_round_one_is_bit_exact` | YES |
| `train_fedseis.py` | NOT PORTED: prototype selection/computation is not wired into `Server` | NOT RUN | NO |
| `train_centralized.py` | NOT PORTED: no centralized replacement is in scope | NOT RUN | NO |
| `train_federated_f3.py` | NOT PORTED: F3/SEG-Y pipeline has no replacement | NOT RUN | NO |
| `train_fedbn_f3.py` | NOT PORTED: F3/SEG-Y pipeline has no replacement | NOT RUN | NO |
| `train_fedprox_f3.py` | NOT PORTED: F3/SEG-Y pipeline has no replacement | NOT RUN | NO |

The base and three algorithm-specific gates use fixed seed 42, BatchNorm,
real seismic data, and compare every state-dict tensor with `torch.equal`.
The Task 3 post-hoc gate also passes. The Task 1 recorded-checkpoint metric
test remains skipped because the matching seed-42 log is not present locally.
