# Migration report

## Task results

| task | result | gate evidence |
|---|---|---|
| Task 1 | DONE with one unavailable local prerequisite | `python3 -m pytest tests/ -q`: `12 passed, 1 skipped`; loss and aggregation gates passed. The recorded-checkpoint metric test skipped because the matching seed-42 log was unavailable. |
| Task 2 | DONE | `tests/test_round_equivalence.py`: `1 passed`; every global state-dict tensor matched with `torch.equal`. |
| Task 3 | DONE | `tests/test_posthoc_equivalence.py`: `1 passed`; reproduced `0.5965` at tau `1.5` and `0.6155` at tau `0.5`. |
| Task 4 | DONE | `MIGRATION_STATUS.md` records four passing round-1 gates and four NOT PORTED entrypoints. |
| Task 5 | SKIPPED by safety gate | `main.py` contains `import algorithms` outside `algorithms/`; the package was retained. |
| Task 6 | DONE for authorized scripts | `train_federated.py`, `train_fedprox.py`, `train_fedbn.py`, and `train_fedvls.py` were deleted after passing round-1 gates. |
| Task 7 | DONE | This report. |

The final post-deletion validation was:

```text
python3 -m pytest tests/ -q
14 passed, 4 skipped
```

The skips are expected retirement-safe legacy checks and the unavailable
matching seed-42 recorded-log check.

## Commits

```text
14f1aab Retire train_fedvls.py (round-1 gate: tests/test_variant_round_equivalence.py)
26dd0de Retire train_fedbn.py (round-1 gate: tests/test_variant_round_equivalence.py)
3c97c29 Retire train_fedprox.py (round-1 gate: tests/test_variant_round_equivalence.py)
3bd3a94 Retire train_federated.py (round-1 gate: tests/test_round_equivalence.py)
93029ac Record migration status for the legacy training scripts
00b55cb Add round-1 equivalence gates for algorithm variants
1fea809 Add post-hoc reproduction gate for the published ensemble numbers
c4218d8 Add round-1 equivalence gate for the federated loop
a6e8c3b Add bit-exact equivalence gates against the legacy scripts
```

## Legacy files

Deleted after passing gates:

- `train_federated.py`: `tests/test_round_equivalence.py`
- `train_fedprox.py`: `tests/test_variant_round_equivalence.py::test_fedprox_round_one_is_bit_exact`
- `train_fedbn.py`: `tests/test_variant_round_equivalence.py::test_fedbn_round_one_is_bit_exact`
- `train_fedvls.py`: `tests/test_variant_round_equivalence.py::test_fedvls_round_one_is_bit_exact`

Retained because no equivalent passed gate exists:

- `train_fedseis.py`: prototype selection and server-side prototype computation are not wired into `Server`.
- `train_centralized.py`: centralized training is outside the replacement scope.
- `train_federated_f3.py`: F3/SEG-Y pipeline is not ported.
- `train_fedbn_f3.py`: F3/SEG-Y pipeline is not ported.
- `train_fedprox_f3.py`: F3/SEG-Y pipeline is not ported.
- `algorithms/`: retained because `main.py` imports it; Task 5 deletion was not authorized.

## Notes

- FedVLS class-frequency discovery was changed to use the dataset index set
  directly, preserving the legacy RNG sequence and passing the round-1 gate.
- The initial round gate divergence was caused by the test harness not setting
  cuDNN deterministic flags; matching the legacy seed setup resolved it.
- No presentation, results, logs, or deck files were intentionally changed.
- The worktree contained unrelated presentation-generation changes while this
  migration ran. They were not reverted, staged, or included here.
