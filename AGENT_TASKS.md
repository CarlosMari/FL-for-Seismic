# Task list for a follow-up agent

Read this whole file before doing anything. Do the tasks **in order**. Do not
skip ahead. Do not do work that is not listed here.

## Ground rules

1. **Never edit a file you have not read in full.**
2. **Never change numbers in `summary.md`, `results.md`, or `presentation.md`.**
   Those are the record of what was measured. If a number looks wrong, write a
   note; do not silently correct it.
3. **Do not start any training run.** No `train_*.py`. Everything here is
   editing, reading, or CPU-only checking. If a task seems to need a GPU, stop
   and report instead.
4. **After every code change, run the verification command given in the task.**
   If it fails, revert your change and report. Do not attempt a second fix.
5. **Do not reformat, rename, or "clean up" anything not named in a task.**
6. If a task's precondition is already true, mark it DONE and move on. Say so.
7. Work on a branch: `git checkout -b agent-fixes`. Commit after each task with
   the message given. **Never force-push. Never touch `master`.**

---

## Task 1 — Fix the vacant-class distillation bug in `train_fedseis.py`

**Why:** when a client is missing exactly one class, the distillation term is
mathematically forced to zero, so the feature silently does nothing.

**File:** `train_fedseis.py`, function `vacant_class_distillation` (~line 160).

**Current code:**
```python
    vc_idx = vacant_mask.nonzero(as_tuple=True)[0]
    local_vc = logits_local[:, vc_idx, :, :]
    global_vc = logits_global[:, vc_idx, :, :]
    local_log_probs = F.log_softmax(local_vc, dim=1)
    global_probs = F.softmax(global_vc, dim=1)
    return F.kl_div(local_log_probs, global_probs, reduction="batchmean")
```

**Replace with** (this is the already-fixed pattern from `train_fedvls.py` —
read that file's `vacant_class_distillation` first and match it):
```python
    vc_idx = vacant_mask.nonzero(as_tuple=True)[0]

    # Softmax over the FULL class dim, then select. Slicing before the softmax
    # makes the |vacant|==1 case degenerate: softmax over a length-1 dim is
    # identically 1.0, so the KL term is structurally zero.
    local_log_probs_full = F.log_softmax(logits_local, dim=1)
    global_probs_full = F.softmax(logits_global, dim=1)

    kl_full = global_probs_full * (
        torch.log(global_probs_full + 1e-10) - local_log_probs_full
    )
    return kl_full[:, vc_idx, :, :].sum(dim=1).mean()
```

**Verify — the KL must be > 0 for a single vacant class:**
```bash
cd /home/carlos/projects/FL-for-Seismic
python3 -c "
import torch, importlib.util
spec = importlib.util.spec_from_file_location('m','train_fedseis.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
lo = torch.randn(2,6,16,16); gl = torch.randn(2,6,16,16)
one = torch.zeros(6, dtype=torch.bool); one[5] = True
two = torch.zeros(6, dtype=torch.bool); two[4] = True; two[5] = True
print('one vacant class :', float(m.vacant_class_distillation(lo,gl,one)))
print('two vacant classes:', float(m.vacant_class_distillation(lo,gl,two)))
"
```
**PASS** = both numbers are non-zero. **FAIL** = the one-class number is 0.0.
If it fails, revert and report.

**Commit:** `Fix vacant-class distillation degeneracy in train_fedseis.py`

---

## Task 2 — Make `logit_suppression` numerically safe

**Why:** `torch.exp` on raw logits overflows to `inf` in float32 past ~88.7,
which produces NaN losses.

**File:** `train_fedvls.py`, function `logit_suppression` (~line 125–143).

Read the whole function first. The loop currently does:
```python
        exp_logit_c = torch.exp(logits_flat[:, c])
        mean_exp = (mask * exp_logit_c).sum() / mask.sum()
        l_c = torch.log(mean_exp + 1e-10)
```

**Replace those three lines with a max-subtracted (log-sum-exp) form:**
```python
        # Max-subtracted: log(mean(exp(x))) == log(mean(exp(x - M))) + M.
        # Without this, exp() overflows to inf for logits above ~88 in float32.
        col = logits_flat[:, c]
        M = col.max().detach()
        mean_exp = (mask * torch.exp(col - M)).sum() / mask.sum()
        l_c = torch.log(mean_exp + 1e-10) + M
```

**Verify — must be finite for large logits, and unchanged for normal ones:**
```bash
cd /home/carlos/projects/FL-for-Seismic
python3 -c "
import torch, importlib.util
spec = importlib.util.spec_from_file_location('m','train_fedvls.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
t = torch.randint(0,6,(2,16,16))
f = torch.ones(6)/6
small = torch.randn(2,6,16,16)
big   = torch.randn(2,6,16,16) + 200.0
a = float(m.logit_suppression(small,t,f)); b = float(m.logit_suppression(big,t,f))
import math
print('normal logits:', a, 'finite:', math.isfinite(a))
print('large logits :', b, 'finite:', math.isfinite(b))
"
```
**PASS** = both finite. **FAIL** = either is `nan` or `inf`.

**Commit:** `Make logit_suppression numerically stable in train_fedvls.py`

---

## Task 3 — Remove one dead line

**File:** `train_federated.py`, in `CrosslineRecallLoss.forward` (~line 178).
Delete the single line `prd_b = preds[b]` — the variable is never read.
**Do not touch anything else in that function.**

**Verify:**
```bash
cd /home/carlos/projects/FL-for-Seismic
python3 -c "import ast;ast.parse(open('train_federated.py').read());print('syntax OK')"
grep -n "prd_b" train_federated.py    # must print nothing
```

**Commit:** `Remove unused variable in CrosslineRecallLoss`

---

## Task 4 — Delete the stale config files

**Why:** `config/fedavg.json` points at `/home/zoe/GhassanGT Dropbox/...` and
CIFAR-10. Neither config is read by any script that produced any result.

**First confirm they are unused. This must print nothing:**
```bash
cd /home/carlos/projects/FL-for-Seismic
grep -rn "config/fedavg.json\|config/fedseismic.json" --include=*.py .
```
If it prints anything, **STOP** and report — they are in use; do not delete.

If it prints nothing:
```bash
git rm config/fedavg.json config/fedseismic.json
```

**Commit:** `Remove unused config files with stale absolute paths`

---

## Task 5 — Pin the environment

**Do not guess versions.** Generate them from what is installed:
```bash
cd /home/carlos/projects/FL-for-Seismic
python3 -m pip freeze > requirements-lock.txt
wc -l requirements-lock.txt
```
Leave the existing `requirements.txt` untouched. Add a line at the top of
`requirements-lock.txt`:
```
# Exact versions captured 2026-08-03 from the machine that produced the results.
```

**Commit:** `Add pinned requirements-lock.txt`

---

## Task 6 — Write unit tests for the two bugs you just fixed

Create `tests/test_losses.py`. Use pytest. **Only test these four things** —
do not invent extra tests:

1. `vacant_class_distillation` (from `train_fedseis.py`) returns > 0 when
   exactly one class is vacant.
2. Same function returns > 0 when two classes are vacant.
3. `logit_suppression` (from `train_fedvls.py`) returns a finite value when
   logits are large (+200).
4. `RecallLossPerSlice` (from `train_federated.py`) returns a finite,
   non-negative scalar on random input.

Load the modules with `importlib.util.spec_from_file_location`, as in the
verification snippets above — these files are scripts, not packages.

**Verify:**
```bash
cd /home/carlos/projects/FL-for-Seismic
python3 -m pytest tests/test_losses.py -v
```
**PASS** = 4 passed. If any fail, report which; do not change the loss code
again to make a test pass.

**Commit:** `Add unit tests for loss edge cases`

---

## Task 7 — Report, do not act

Write `AGENT_REPORT.md` in the repo root containing:

- One line per task: DONE / SKIPPED / FAILED, with the verification output.
- The exact `git log --oneline` of the commits you made.
- Anything you noticed but did **not** change.

Then stop. **Do not open a pull request. Do not merge. Do not touch `master`.**

---

## Explicitly OUT OF SCOPE — do not do these

- **Do not** refactor duplicated code across `train_*.py`. Those scripts
  produced published results; a refactor risks changing them silently. This
  needs a human decision, not an agent.
- **Do not** change `algorithms/fedseismic/metrics.py`. Its `average='weighted'`
  metric is a known issue, already documented in `plan.md` §0.1. Changing it
  now would make old logs incomparable to new ones.
- **Do not** change checkpoint-selection logic in `train_federated.py` or
  `train_centralized.py`. The test-set leak is real and documented, but the
  fix changes every reported number and must be a human call.
- **Do not** move `run_*.sh` or the `logs_*/` directories. Paths are referenced
  from figure scripts and notes.
- **Do not** re-run any experiment, regenerate any figure, or rebuild the
  PowerPoint.
- **Do not** edit `presentation.md`, `plan.md`, `summary.md`, or `results.md`.
