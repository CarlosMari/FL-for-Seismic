# Migration tasks — verify PR #3, then retire the legacy scripts

Read this whole file first. Do the tasks **in order**. Do not skip ahead.
Do not do anything not listed here.

## Context

PR #3 added `fedseismic/` (1,723 lines, 28 files, **0 deletions**). Nothing has
been removed yet, and that is correct: the legacy scripts must stay until each
replacement has passed its gate.

Already verified by hand (not by the test suite):
- All **10 losses** are bit-exact against `train_federated.py`, including
  `recall_slice`.
- All **8 aggregators** are registered.
- `pytest tests/ -q` → **10 passed**.

**The problem:** only `tests/test_partition.py` actually compares against the
legacy code. The other tests check the new code against itself. The plan's
central requirement — bit-exact equivalence gates — is mostly *not* committed.
Task 1 fixes that. Nothing may be deleted before it passes.

## Ground rules

1. **Never delete a legacy file until its gate passes.** No exceptions.
2. **One task per PR.** Base every PR on `experiments/rare-class-fl-image2026`.
3. **A gate failure means STOP and report.** Never weaken a gate to make it pass.
4. **No behaviour changes disguised as migration.** Found a bug? Note it, keep
   the buggy behaviour, fix it in a separate PR afterwards.
5. **Do not touch**: `presentation.md`, `build_presentation.py`, `figures/`,
   `*.pptx`, `summary.md`, `results.md`, `plan.md`, anything under `results/`
   or `logs_*/`.
6. **No training runs** except the explicit 1-round gates below (~2 min each).
7. `pytest` is installed (9.1.1). Run it; do not hand-run tests.
8. Work on a branch per task. Never force-push. Never touch `master`.

---

## Task 1 — Commit the equivalence gates that are missing ⚠️ BLOCKING

**Why:** these gates are the only thing that justifies deleting anything later.
They currently exist only as throwaway scripts.

Create `tests/test_legacy_equivalence.py`. Load the legacy modules with
`importlib.util.spec_from_file_location` (they are scripts, not packages).
**Skip, do not fail, if a legacy file is absent** — so the suite still passes
after Task 6 deletes them:

```python
import importlib.util, pathlib, pytest, torch

ROOT = pathlib.Path(__file__).resolve().parents[1]

def _legacy(name):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} already retired")
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
```

Write these tests:

1. **All 10 losses bit-exact.** Fixed `torch.manual_seed(0)`, logits
   `(3,6,32,32)`, targets `randint(0,6,(3,32,32))`. Assert `float(old(...)) ==
   float(new(...))` exactly — not `approx`. `UnifiedFocalLoss` and
   `AsymmetricTverskyLoss` need `alpha_per_class` / `beta_per_class`; pass
   `[0.5]*6` for both.
2. **All 8 aggregators bit-exact.** Fixed state dicts and fixed client class
   stats. Compare returned weight lists elementwise. `accuracy` needs a model
   and loader — if that is impractical, `pytest.skip` it with a comment saying
   why, and cover the other 7.
3. **Evaluation matches a real recorded run.** Load
   `results/fedavg_noniid_20c_20r_sr0.25_agginvfreq_miou_lossrecall_slice_frc_s42/global_round_19.pth`,
   evaluate with `fedseismic.eval.metrics`, and assert the macro mIoU equals
   the round-20 value in
   `logs_phase_v/V4_v3_frc_s99.log`-style logs for the matching seed, to 4 dp.
   If the checkpoint is missing, `pytest.skip`.

**Gate:** `python3 -m pytest tests/ -q` → all pass, none of the three new tests
skipped on this machine.

**Commit:** `Add bit-exact equivalence gates against the legacy scripts`

---

## Task 2 — Phase 5 gate: one federated round must match

**Why:** the training loop is the one part where a silent divergence would
change results, and bistability means a full-run comparison proves nothing.

Add `tests/test_round_equivalence.py`:

- Fixed seed (42), `--num_rounds 1`, `--num_clients 20`, `--sample_ratio 0.25`,
  `--loss recall_slice`, `--agg_strategy equal`, BatchNorm.
- Run one round through `fedseismic.federated.Server` and one through the
  legacy path in `train_federated.py`.
- Assert **every tensor** in the resulting global state dict is bit-identical
  (`torch.equal`), not just the mIoU.

If the legacy path cannot be driven in-process without a refactor, instead:
run `python3 train_federated.py ... --num_rounds 1 --seed 42` and the
`fedseismic` CLI equivalent as subprocesses, then compare the two saved
`global_round_0.pth` files tensor by tensor. Write results to a scratch dir,
**not** into `results/`.

**Gate:** every tensor identical. If any differ, STOP and report which layers
and the max absolute difference. **Do not proceed to Task 3.**

**Commit:** `Add round-1 equivalence gate for the federated loop`

---

## Task 3 — Phase 6 gate: reproduce the published post-hoc numbers

Using `fedseismic.eval.posthoc`, reproduce **both** rows from
`paper_figures/ensemble_v3_frc_final.txt` and `..._best.txt`:

| checkpoint | tau | expected avg mIoU |
|---|---|---|
| `global_round_19.pth` | 1.50 | **0.5965** |
| `best_global_model.pth` | 0.50 | **0.6155** |

Seeds `[42, 123, 7, 99, 2025]`, config
`fedavg_noniid_20c_20r_sr0.25_agginvfreq_miou_lossrecall_slice_frc_s{seed}`.

Put this in `tests/test_posthoc_equivalence.py`, tolerance 1e-4, and
`pytest.skip` if the checkpoints are absent.

**Gate:** both values reproduce. These are numbers in the talk — if they do not
reproduce, STOP and report.

**Commit:** `Add post-hoc reproduction gate for the published ensemble numbers`

---

## Task 4 — Wire the legacy entrypoints to the new package

**Do not delete anything yet.** For each of `train_fedprox.py`,
`train_fedbn.py`, `train_fedvls.py`, `train_fedseis.py`: confirm whether an
equivalent `ClientTrainer` subclass exists in `fedseismic/federated/`.

Write the answer into `MIGRATION_STATUS.md` as a table:

| legacy script | fedseismic equivalent | round-1 gate | safe to delete |
|---|---|---|---|

Add a round-1 equivalence test (as Task 2) for **each** script that has an
equivalent. For any script with no equivalent, mark it `NOT PORTED` — it stays.

**Gate:** `MIGRATION_STATUS.md` exists and every row is filled with evidence,
not assumption.

**Commit:** `Record migration status for the legacy training scripts`

---

## Task 5 — Delete the dead framework

`algorithms/` produced **no published result**. Confirm before deleting:

```bash
grep -rn "from algorithms\|import algorithms" --include=*.py . \
  | grep -v "^./algorithms/"
```
If that prints anything outside `algorithms/` itself, **STOP** — it is in use.

If it prints nothing:
```bash
git rm -r algorithms/
```
`config/` was already removed in PR #2.

**Gate:** `python3 -m pytest tests/ -q` still passes, and
`python3 -c "import fedseismic"` works.

**Commit:** `Remove the unused algorithms package`

---

## Task 6 — Retire legacy scripts, one commit each

**Only** for scripts whose round-1 gate passed in Task 2 or Task 4.

For each, in a separate commit:
1. `git rm <script>`
2. Commit message must name the gate that authorised it, e.g.
   `Retire train_federated.py (round-1 gate: tests/test_round_equivalence.py)`
3. Run `python3 -m pytest tests/ -q` after each deletion. The legacy
   equivalence tests should now **skip**, not fail — that is why Task 1 used
   `pytest.skip`.

**Do not delete** any script marked `NOT PORTED` in Task 4.

**Gate:** suite green after every individual deletion.

---

## Task 7 — Report

Write `MIGRATION_REPORT.md`:
- One line per task: DONE / SKIPPED / FAILED, with the gate output pasted.
- `git log --oneline` of every commit made.
- Every legacy script: deleted (with its gate) or retained (with the reason).
- Anything noticed but not changed.

Then stop. Do not merge anything to `master`.

---

## Out of scope — do not do these

- **Do not** change `algorithms/fedseismic/metrics.py` before Task 5 deletes it.
  Its `average='weighted'` metric is a known issue; the new package already
  uses macro only.
- **Do not** "fix" the best-round checkpoint selection in the legacy scripts.
  It is documented, and the new `CheckpointPolicy` already defaults to FINAL.
- **Do not** re-run any experiment, regenerate any figure, or rebuild the deck.
- **Do not** rename or move `logs_*/`, `results/`, or `paper_figures/`.
- **Do not** add features to `fedseismic/`. This is migration, not development.
