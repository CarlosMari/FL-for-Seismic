# Refactor plan — `fedseismic/`: one framework for bistable segmentation FL

**Goal.** Replace seven duplicated `train_*.py` scripts and the half-finished
CIFAR-era `algorithms/` package with a single library shaped for *this* problem:
geographic non-IID seismic segmentation where the outcome is **bistable**.

**Non-goal.** Reproducing old numbers exactly. Bistability makes that impossible
(see §0.3). The contract is *equivalence of the deterministic parts*, not
equality of end results.

**Status of the code being replaced.** `train_federated.py` (all IMAGE results),
`train_fedseis.py`, `train_fedvls.py`, `train_fedprox.py`, `train_fedbn.py`,
`train_centralized.py`, `train_federated_f3.py`, `train_fedbn_f3.py`,
`train_fedprox_f3.py`, plus `algorithms/` (produced no published result).

---

## 0. Preconditions — do these before touching any code

### 0.1 Commit everything currently uncommitted ⚠️ BLOCKING

The premise "worst case it's all in git" is **currently false**. Uncommitted:

```
 M ensemble_posthoc.py          (final/best checkpoint switch)
 M summary.md                   (Tier 0 re-analysis)
 M train_federated.py           (UNET_NORM tag)
 M train_tools/models/unet_parts.py  (GroupNorm support)
?? presentation.md, build_presentation.py, figures/slide_charts.py,
   figures/slides/, IMAGE2026_FL_Seismic.pptx, run_tier01.sh, AGENT_TASKS.md
```

Commit all of it to `experiments/rare-class-fl-image2026` and push **before
anything else**. `logs_tier01/` and `results/` stay local by decision — but
then they are *not* backed up, so copy them to external storage too.

### 0.2 Tag the pre-refactor state

```bash
git tag pre-refactor-image2026
git push origin pre-refactor-image2026
```
This is the rollback point. Every claim in the talk is reproducible from it.

### 0.3 Understand the verification limit

**You cannot verify this refactor by re-running and comparing final mIoU.**
10 of 14 configs produce both outcomes (C5 ≈ 0.25 *or* exactly 0.000) depending
on seed alone. A refactored run landing at 0.000 tells you nothing about
correctness.

Therefore the verification contract is:

| Component | Verifiable how |
|---|---|
| Losses | bit-identical output, fixed input tensors |
| Aggregation | bit-identical weights, fixed state dicts |
| Partitioning | identical index lists |
| Evaluation | identical mIoU on a fixed checkpoint |
| Local training | identical weights after **1 round**, fixed seed |
| Full run | **NOT verifiable** — do not attempt |

### 0.4 Freeze the talk

Do not touch `presentation.md`, `build_presentation.py`, `figures/`, or the
`.pptx` during this work. IMAGE is ~2 weeks out. If the refactor is unfinished
by then, that is fine — the talk depends on the tag from §0.2, not on this.

---

## 1. Target layout

```
fedseismic/
  __init__.py
  config.py          # RunConfig dataclass; no JSON, no argparse sprawl
  data/
    partition.py     # geographic + iid splits, client class statistics
    seismic.py       # cube loading, normalisation, InlineLoader
  models/
    unet.py          # moved from train_tools/models/, norm selectable
  losses/
    __init__.py      # LOSSES registry: name -> class
    focal.py dice.py tversky.py recall.py fciou.py
  federated/
    server.py        # Server: round loop, sampling, aggregation, eval
    client.py        # ClientTrainer: download / train / upload / reset
    aggregation.py   # AGGREGATORS registry: 8 strategies
    sampling.py      # uniform, force_rare_client
  eval/
    metrics.py       # ONE macro-mIoU implementation. No 'weighted'.
    posthoc.py       # logit adjustment (tau), seed ensembling
  experiment.py      # runs n seeds, returns SeedResults
  cli.py             # `python -m fedseismic.cli run --config ...`
tests/
  test_losses.py test_aggregation.py test_partition.py
  test_metrics.py test_equivalence.py
```

Deleted at the end: `algorithms/`, `config/`, all `train_*.py`, `measures.py`.

---

## 2. Design decisions that come from the findings

These are the reason to do this at all — they encode what we learned.

### 2.1 A run is n seeds, never one

`experiment.run(cfg, seeds=[42,123,7,99,2025])` returns:

```python
@dataclass
class SeedResults:
    miou_final: list[float]      # per seed, ROUND 20
    miou_best:  list[float]      # per seed, best round (bias-prone)
    c5_final:   list[float]
    history:    list[list[RoundRecord]]   # every round, every seed

    @property
    def mean_std(self) -> tuple[float, float]: ...
    @property
    def recovery_rate(self) -> float:
        """Fraction of seeds with final rare-class IoU > 0.15.
        THE headline metric for a bistable process."""
```

No API returns a bare float for a config. That single choice makes the
best-round mistake structurally hard to repeat.

### 2.2 Checkpoint policy is explicit and defaults to honest

```python
class CheckpointPolicy(Enum):
    FINAL = "final"      # default
    BEST_VAL = "best_val"
    BEST_TEST = "best_test"   # logs a warning: leaks the eval set
```

`BEST_TEST` stays available — old numbers must remain reproducible — but it
warns, and `SeedResults` always carries both.

### 2.3 One metric

`eval/metrics.py` exposes exactly one function returning `(macro_miou,
per_class_iou)`. `average='weighted'` does not exist in the new code. This is
the `metrics.py:39` discrepancy, removed by construction.

### 2.4 Registries, not if-chains

```python
LOSSES = {"focaldice": FocalDiceLoss, "recall_slice": RecallLossPerSlice, ...}
AGGREGATORS = {"equal": EqualAgg, "invfreq_miou": InvFreqMiouAgg, ...}
```
Adding a variant = one class + one registry line, no edits to the loop.

---

## 3. Execution order

Each phase ends green before the next begins. Branch: `refactor/fedseismic`.

### Phase 1 — Losses (safest, largest duplication)
Move all 10 loss classes from `train_federated.py` into `fedseismic/losses/`.
Copy verbatim first; refactor only after tests pass.

**Gate:** `tests/test_losses.py` asserts, for each loss, that
`new(logits, targets) == old(logits, targets)` bit-exactly on fixed random
tensors (seeded). Import the old ones from the tagged commit.
Include the `|vacant|==1` and large-logit edge cases already fixed in PR #2.

### Phase 2 — Data & partitioning
Move `partition_noniid`, `partition_iid`, `build_client_loaders`,
`load_and_normalize`, `compute_client_class_info`, `InlineLoader`.

**Gate:** identical partition index lists for n_clients ∈ {3,5,20}, both splits.
Identical client class-fraction arrays.

### Phase 3 — Evaluation
One `evaluate(model, loader, labels) -> (macro_miou, per_class)`.

**Gate:** on a fixed existing checkpoint from `results/`, the new evaluator
returns the mIoU recorded in that run's log, to 4 decimals.

### Phase 4 — Aggregation & sampling
Port all 8 strategies + `force_rare_client`.

**Gate:** given fixed state dicts and fixed client stats, new aggregation
returns bit-identical weights to the old function for every strategy.

### Phase 5 — Server / Client loop
Rewrite `BaseServer`/`BaseClientTrainer` for segmentation. Keep the method
shape; drop `from .measures import *`, the `dataset != 'seismic'` branches,
and all accuracy/classwise CIFAR bookkeeping.

**Gate:** a 1-round run with a fixed seed produces weights bit-identical to
`train_federated.py` at 1 round. **Do not compare beyond round 1.**

### Phase 6 — Post-hoc
Port logit adjustment (tau sweep) and seed ensembling from
`metafusion_posthoc.py` / `ensemble_posthoc.py`. Remove the hardcoded
`/home/carlos/...` paths.

**Gate:** reproduce `paper_figures/ensemble_v3_frc_final.txt` exactly
(0.5965 at tau=1.5) from the same checkpoints.

### Phase 7 — Algorithm variants
FedProx, FedBN, FedVLS as `ClientTrainer` subclasses. FedVLS keeps the fixed
vacant-class distillation from PR #2.

**Gate:** 1-round equivalence against each old script, as Phase 5.

### Phase 8 — Delete
Only now: remove `algorithms/`, `config/`, and every `train_*.py` whose
replacement passed its gate. One commit per deletion, message naming the gate
that authorised it.

---

## 4. Rules for whoever executes this

1. **One phase per PR.** Never combine.
2. **Copy verbatim, then refactor.** Two separate commits, always.
3. **A phase is not done until its gate passes.** If a gate fails, stop and
   report — do not weaken the gate.
4. **Never delete an old file in the same PR that adds its replacement.**
5. **No behaviour changes disguised as refactoring.** Found a bug mid-port?
   Note it, port the buggy behaviour, fix it in a separate follow-up PR.
6. **Do not touch** `presentation.md`, `build_presentation.py`, `figures/`,
   `*.pptx`, `summary.md`, `results.md`, or anything under `results/`.
7. **No training runs** except the 1-round equivalence checks (~2 min each).

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| Silent behaviour change | Bit-exact gates on all deterministic parts |
| Bistability masks a bug | Never compare full runs; round-1 only |
| Scope creep into the talk | §0.4 freeze; separate branch |
| Half-finished at IMAGE | Tag from §0.2 is the talk's source of truth |
| Losing old numbers | `BEST_TEST` policy retained, warned not removed |

---

## 6. What this buys

- One implementation instead of ten.
- The bistability finding encoded in the API: n seeds, recovery rate, explicit
  checkpoint policy, one metric.
- `results/` grows a machine-readable history instead of regex-parsed logs.
- A base the next paper can extend by adding one class.
