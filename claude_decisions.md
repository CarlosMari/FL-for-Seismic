# FedHCA2-style experiment: decisions

Scope: compare a FedHCA2-style method with baselines on F3, Parihaka and
Penobscot, one client per survey. Code is prepared and tested on CPU with
synthetic data. **Nothing has been trained on the real surveys, so there are no
results yet.** Upstream means github.com/innovator-zero/FedHCA2 (Lu et al.,
CVPR 2024), read from its `aggregate.py`, `models/hyperweight.py`, `train.py`
and `configs/pascal_st_nyud_mt_1c.yml`.

## Method: what was kept from FedHCA2

| Decision | Why |
|---|---|
| Treat one survey as one "task" | Upstream federates different tasks. Here every client does segmentation, so the survey is the unit that conflicts. This is an adaptation, so call it "FedHCA2-style", not FedHCA2. |
| Keep conflict-averse aggregation for the encoder (`inc`, `down1-4`) | It is the part of the method aimed at conflicting client updates, and it does not depend on tasks differing. |
| Keep cross-attention aggregation for the decoder (`up1-4`), per layer | Same reasoning. All UNet decoders have the same shape, so it applies directly. |
| Keep learned per-client `alpha` and per-layer, per-client `beta`, clamped to [0, 1] | They are what make the update personal. Every client ends each round with its own encoder and decoder. |
| Drop multi-task clients, per-task decoder groups and task loss weights | All clients have one task. |
| Drop homo-client averaging | With one client per survey it does nothing. Runs with more than one client per survey are refused instead of silently behaving differently from upstream. |

## Disjoint label spaces

| Decision | Why |
|---|---|
| `outc` (head) is local and never aggregated | Your request. The surveys share no classes, so averaging heads has no meaning. |
| BatchNorm is local (FedBN) | Your request. It also matches the 2026-10-29 finding that local BN closes the BN re-identification channel. |
| Keep one 19-row union head (F3 0-5, Parihaka 6-11, Penobscot 12-18) | This is the repo's existing convention, and `rescore_run` and the checkpoints rely on it. |
| Mask the loss to each client's own rows, and set the others to -1e9 in the deployed model | Reused from `FedPerSurveyBNClientTrainer`, so FedHCA2 and fedpersurvey differ only in aggregation. |
| Penobscot overburden stays ignored (`survey_c_ignore: 0`) | Existing convention. |
| `reconstruct` stays local | Nothing trains it, so it should not take part in aggregation. |

## Choices that differ from upstream

| Decision | Why | Risk |
|---|---|---|
| Local training keeps the repo's settings: AdamW, lr 1e-3, 3 local epochs, 20 rounds, optimizer rebuilt each round, no scheduler | The comparison with the repo's baselines must differ only in aggregation. Upstream uses lr 1e-4 with a cosine schedule and warmup. | The method's defaults were not tuned for this learning rate. |
| Hyperweight defaults are upstream's: `ca_c` 0.4, `alpha`/`beta` init 0.1, SGD lr 0.01, momentum 0.9, weight decay 1e-4 | No tuning data exists yet, and tuning on the holdout would bias the comparison. | Hyper-gradients are dot products of raw weight deltas, so they scale with delta size. With this larger local lr, the weights may saturate at 0 or 1. Check `hyper_weights.jsonl` after seed 0. |
| Hyper-gradients are written analytically, not by backprop | Both updates are linear in their weight, so the result is the same without holding a graph over about 31M parameters. A test compares them with autograd, to 2 decimal places. | |
| Gram matrix is divided by its mean before SLSQP | The optimum is unchanged, but the solver's absolute tolerance no longer depends on delta size. | |
| Aggregation runs on CPU | The GPU is the scarce resource, and the parity OOM showed how little margin there is. | Slower rounds, not measured on real data. |

## Server integration

| Decision | Why |
|---|---|
| New module `federated/fedhca2.py` plus one branch in `Server.run_round` | Same shape as the other algorithms. The server normally assumes a single global model, so this is the smallest hook that allows per-client models. |
| The server's global model is kept as a placeholder | Logging and checkpoint code expect one. Its body is a plain average and its head and BN are never trained. |
| Algorithm added to the personal-model lists in `multisurvey_utility.py` | Rescoring must use each client's deployed model, and no global reference is reported, same as fedpersurvey. |
| Personal checkpoints are rewritten after aggregation | The deployed model is the post-aggregation one, matching upstream's evaluation order. |
| Config validation refuses `survey_solo`, `sample_ratio` below 1, non-multisurvey runs and more than one client per survey | The aggregator needs every client every round. |

**Do not read `final`, `best`, `c5` or `global_on_local` in a fedhca2 `results.json`, or `global_acc` in its history.** They come from the placeholder model. Use the tables from the comparison script.

## Experiment design

| Decision | Why |
|---|---|
| Arms: solo x3, fedpersurvey, fedhca2 | Solo answers "does federating help". fedpersurvey at n=3 is exactly FedAvg body + local head + FedBN, so it isolates the effect of the new aggregation. |
| No FedAvg or FedBN arm | Both share a head across disjoint label spaces, which you ruled out. |
| Solo uses `fedpersurvey` with `num_clients: 1` | Same masked loss and code path as the FL arms, and one client gets the same train indices as in the FL arms. |
| New baselines instead of reusing `multisurvey3_fedpersurvey_rot` or `multisurvey_solo_k6_*` | Those use 6 clients or 2 per survey, and the 3-survey config has no spatial buffer, so they are not like-for-like. |
| `spatial_holdout_buffer: 50` | `seismic_multisurvey3_rot.json` lacks it, which would inflate holdout scores. |
| `batch_size: 2` | Parihaka and Penobscot sections are large, and parity FL ran out of memory at batch 4. Used in every arm. |
| Corrected `*_rot90` Parihaka | Required by the 2026-10-29 rotation fix. |
| Full surveys, no `survey_train_subset` | You asked for the three datasets as they are. Survey sizes therefore differ, as in the size-matched parity discussion, so do not read results as parity. |
| Seed 0 pilot first | The prepared sweep files specify seeds 0, 1 and 2, but `run_multisurvey3_k3_seed0.sh` overrides them to seed 0 only. Review the completed results with the advisor before deciding whether to run other seeds. |
| Primary metric: survey-block all-class mIoU on each buffered holdout, with present-class also reported | Compute the mean over the survey's own 6/6/7 classes, including absent classes as zero. The lower-level `multisurvey_utility.py` also reports a 19-row union mean; retain that separately as `miou_union_all_class`, not as the survey's all-class score. |
| FL minus solo is paired by survey and seed | Same convention as the parity analysis. |
| No official F3 test score | Only F3 has one, so it cannot compare across surveys. |
| New output names (`multisurvey3_k3_*`) | Avoids overwriting `multisurvey_utility.csv`, which has uncommitted changes. |

## Process

| Decision | Why |
|---|---|
| Nothing was launched, and all tests ran with `CUDA_VISIBLE_DEVICES=""` | You said the GPU is busy. |
| Skipped the design and spec steps | You said "just prepare the code". |
| Nothing was committed | You did not ask. Runs record `git_head` and `git_dirty`, and this repo treats uncommitted-code results as invalid, so commit before running. |
| No Claude trailers | Your standing preference, for when you do commit. |
| Unrelated uncommitted files were left alone | `multisurvey_fedavg_rot.json`, `multisurvey_utility.csv`, the `.pyc` file and the `parity_*_s1.json` sweeps are not part of this work. |

## How to run

```bash
bash scripts/run_multisurvey3_k3_seed0.sh [parity-analysis-parent-pid]
```

This runs the five seed-0 arms in sequence, skips finished seeds, then writes
`results/tables/multisurvey3_k3_{runs,summary,per_class}.csv`. The original
`run_multisurvey3_k3.sh` still runs all 15 jobs; do not use it for the pilot.
