# Metrics

Single reference for scores produced by `fedseismic`. Field names in
`RoundRecord` and `results.json` still say `miou_*` on classification tasks.
The primary score is accuracy on BloodMNIST, OrganCMNIST, and CIFAR-10, and
macro mIoU on seismic (`score_loader` in `fedseismic/eval/metrics.py`).

π is a client's true label histogram: class fractions on its **train** split.
π̂ is an attacker's estimate of that histogram. Distances use histograms
renormalized onto the simplex (`to_simplex`).

The boundary figure (`results/boundary_logit.png`) is one point per method,
seed-averaged. There are no error bars and no lines. Upload and oracle FedKPer
runs are omitted. FedBN is omitted.

- **X** is last-upload leakage advantage. Higher means the upload reveals more
  of π than the global model that client just received.
- **Y** is mean local holdout accuracy. Higher is better.

The older CLI Pareto (`python -m fedseismic.cli plot`) still plots raw softmax
leakage against `personalization_gap`. That is not the figure we use.

---

## Utility

### Global test (`global_score`)

**How.** After the last round, score the aggregated model on the official test
set. Classification: example accuracy. Segmentation: macro mIoU. Seismic cubes
with `test1` and `test2` average those two into `miou_final`.

**Means.** Shared-model quality on the population the papers quote. Higher is
better. This is not the y-axis. On a 20-client Dirichlet split it can be far
below local holdout accuracy, because each client's test is mostly its own
majority classes.

### `local_mean`

**How.** For every client that trained, score its **deployed** local weights
on its own holdout (`local_test_ratio`, default 0.2, disjoint from its train
indices). Average those scores. FedPer and Ditto are scored from
`personal_state` in memory. The `client_XXX.pt` on disk is the upload, which
for FedPer has the global classifier put back. FedLC adds `log π` again at
this local score only.

**Means.** Typical personalized accuracy. This is the y-axis. Higher is better.

### `local_worst`

**How.** Minimum of the same per-client local scores.

**Means.** The most disadvantaged client. Higher is better. Not on the current
boundary figure. A separate worst-10% check pools the bottom clients rather
than taking the single minimum.

### `global_on_local`

**How.** Score the final aggregated model on every client's holdout, then
average. No logit adjustment.

**Means.** How useful the shared model is on client-specific data.

### `personalization_gap`

**How.** `local_mean - global_on_local`.

**Means.** Extra accuracy from the deployed local model on the same holdouts.
This can shrink when the shared model gets better and `local_mean` stays flat.
Report `local_mean` beside it. It is not the plot y-axis.

### Classwise holdout accuracy

**How.** On each client's holdout, count correct predictions per class for the
saved local checkpoint and for `global_final.pt`. Pool counts across clients
and seeds. A class absent from that holdout does not enter. No `log π` is
added, so this understates deployed FedLC. Saved FedPer and Ditto checkpoints
are uploads, so their private heads are not in this table.

**Means.** Where the local model is ahead. The gain so far sits on classes the
global model misses.

---

## Leakage advantage (plot x-axis)

Each round, for each participating client, the server logs the mean softmax of
the **upload** and of the **global weights that client just downloaded**, on a
public probe. The probe is `test1` (CIFAR: the official test set), shuffle
off. FedKPer `infer` reads batches `0 .. privacy_probe_batches-1` (default 8).
The logged attack uses the next `privacy_round_batches` batches (16), so it
does not reuse the aggregation probe.

```
advantage = (1 - TV(π, π̂_upload)) - (1 - TV(π, π̂_global))
TV(π, π̂) = (1/2) Σ_k |π_k - π̂_k|
```

`last_advantage` is each client's last participation, then the mean over
clients. `mean_advantage` averages every participation of that client first,
then averages clients. The plot uses `last_advantage`.

**Means.** How much closer the upload's class mix on public data is to the
private train mix, beyond what the global model already shows. Higher is worse
privacy. Near zero means the upload is no more informative than the global
model on this attack. It can be slightly negative.

This is not raw leakage. Raw `leakage = 1 - TV` stays in the sweep CSV as the
softmax column. A high raw leakage with a low advantage means the global model
already carried the mix.

### Presence

**How.** A class is present if its train mass is above `1e-3`. The score is
the residual `π̂_upload,k - π̂_global,k`. Rows are pooled over clients, rounds,
and seeds, so they are not independent samples. AUC is the ranking of present
vs absent. TPR at 1% FPR is read off that ROC.

**Means.** Whether the server can tell which classes the client has, rather
than the full proportions. Histogram advantage can fall while this rises.
Presence F1 on a softmax that almost never predicts absence is not a result;
the predicted-present rate has to be reported with it.

### Gradient inversion (PSNR)

**How.** One labeled training image. The attacker is given the true one-step
gradient of the training loss and the label. A dummy starts from `randn` and
is optimized to match that gradient in cosine similarity, plus total variation
(Geiping et al., NeurIPS 2020). FedKPer includes the same KL term. FedPer drops
the classifier parameters from both gradients. PSNR is on pixels mapped back
to `[0, 1]`. A constant image is the chance floor, because these datasets are
smooth.

**Means.** Whether one gradient reveals the image. Higher PSNR is a stronger
attack. This is not the five-epoch weight delta the server actually receives.

### Membership (`mia_auc_local`, `mia_auc_global`)

**How.** Up to 64 of a client's train examples versus up to 64 of that same
client's holdout. Score by true-class confidence and by negative loss. AUC
near 0.5 is chance.

**Means.** Example membership, not the class mix. Appendix only.

---

## Other estimators (not the plot x-axis)

Same `{tv, kl, leakage, presence_f1}` suffixes in the sweep CSV.

- **Softmax prior.** Mean softmax of the uploaded model on the public probe.
  `leakage` in the CSV is this one.
- **Global softmax.** The same probe through the aggregated model, still
  compared with that client's π.
- **Last-layer prior.** Softmax of last-layer row norms (plus bias). No data.
- **Meta / weight delta.** Ridge from a sketch of `(local - global)` to π,
  leave one client out. Uses other clients' true histograms as targets, so it
  is an optimistic server.
- **KL.** `Σ π log(π / π̂)` after smoothing. Lower KL is a stronger attack.
  The boundary figure does not use it.

FedKPer aggregation weight, only in `infer` / `upload` / `oracle` mode:

```
w_i = (client train accuracy)_i × (ε + normalized_entropy(π̂_i))
```

`oracle` uses true π and `upload` uses a client-sent π. Those two are not
attacks, and they are not on the figure.

---

## Sweep columns

| Column | Read it as |
|---|---|
| `local_mean` | Deployed local model on client holdouts. Plot y. |
| `global_score` | Aggregated model on the official test set. |
| `global_on_local` | Aggregated model on client holdouts. |
| `personalization_gap` | `local_mean - global_on_local`. Not the plot y. |
| `leakage` | Softmax `1 - TV` of the upload. Not the plot x. |
| last-upload advantage | Upload leakage minus the global the client received. Plot x. From `round_scores.jsonl`, not a sweep column. |

Code: `fedseismic/eval/metrics.py`, `fedseismic/privacy/metrics.py`,
`fedseismic/privacy/round_scores.py`, `fedseismic/federated/server.py`
(`_log_round_scores`, `evaluate_personalized_local`).
