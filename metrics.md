# Metrics

Definitions for the scores in `findings.md`. Field names in `RoundRecord` and
`results.json` still say `miou_*` on classification. The number is accuracy
on BloodMNIST, OrganCMNIST, and CIFAR-10, and macro mIoU on seismic
(`score_loader` in `fedseismic/eval/metrics.py`).

π is a client's true label histogram on its train split. π̂ is an attacker's
estimate. Distances use histograms on the simplex (`to_simplex`).

The figure is `results/figures/boundary_r2.png`, written by
`python -m fedseismic.privacy.boundary_r2`. One point per method, mean of
seeds 0, 1, and 42. No error bars and no lines. Upload and oracle FedKPer
runs are omitted. FedBN is omitted. The top row is the last-upload softmax.
The bottom row is Whispers of Data.

- **X** is class-mix R². Higher means the attacker names the client's class
  fractions. 1 is a perfect mix. 0 is the score of predicting every client's
  average fraction. Negative is worse than that average.
- **Y** is mean local holdout accuracy of the deployed model. Higher is better.

R² is `sklearn.metrics.r2_score` on the predicted and true fractions, classes
pooled. It is not the square of Pearson. Pearson is reported separately. A
negative R² with Pearson near 0 is a flat miss. A positive Pearson with R²
near 0 is a real correlation whose slope is too flat to track the fractions.

The older advantage plot (`results/figures/boundary_logit.png`) uses
last-upload total-variation advantage as x. The CLI Pareto
(`python -m fedseismic.cli plot`) still plots raw softmax leakage against
`personalization_gap`. Neither is the current figure.

---

## Utility

### Global test (`global_score`)

**How.** After the last round, score the aggregated model on the official
test set. Classification: example accuracy. Segmentation: macro mIoU.

**Means.** Shared-model quality on the population papers quote. This is not
the y-axis. On a 20-client Dirichlet split it can sit far below local holdout
accuracy, because each client's holdout is mostly its own majority classes.

### `local_mean`

**How.** For every client that trained, score its deployed weights on its own
holdout (`local_test_ratio`, default 0.2, disjoint from its train indices).
Average those client scores. FedPer, FedBABU, and FedRoD are scored from the
saved personal head. The upload file for FedPer has the global classifier put
back. FedLC adds `log π` again at this local score only. FedBABU's reported
value is the classifier fit after the last round. FedRoD's is the sum of the
generic head and the personal residual.

**Means.** Typical personalized accuracy. This is the y-axis. Higher is better.
It is a mean of clients, so a large client does not dominate. It rewards
majority classes inside each client.

### `local_worst`

**How.** Minimum of the same per-client local scores.

**Means.** The most disadvantaged client. Higher is better. Not on the current
figure.

### `global_on_local`

**How.** Score the final aggregated model on every client's holdout, then
average. No logit adjustment, unless a sentence says otherwise.

**Means.** How useful the shared model is on client-specific data.

### `personalization_gap`

**How.** `local_mean - global_on_local`.

**Means.** Extra accuracy from the deployed local model. It shrinks when the
shared model improves and `local_mean` stays flat. It is not the plot y-axis.

### Macro-F1

**How.** On each client's holdout, F1 over classes with train mass above
`1e-3`, then the mean over clients. FedLC adds `log π` on the local model
only. FedPer, FedBABU, and FedRoD use the saved personal head
(`fedseismic/privacy/classwise.py`).

**Means.** Accuracy on the classes the client actually has. Mean accuracy can
call a seed a win when this is a tie.

### Classwise holdout accuracy

**How.** Pool correct predictions per class across clients. A class absent
from a holdout does not enter that count.

**Means.** Which classes carry the local-model gain. On Blood those are the
classes the shared model misses.

---

## Class-mix R² (plot x-axis)

**How.** Take each client's true train fraction and the attacker's predicted
fraction. Concatenate across classes and clients. Score with `r2_score`.
The softmax row uses the last participation's public softmax of the upload.
The Whispers row uses the mix predicted from that client's per-class accuracy
trajectory.

**Means.** Whether the attacker recovers the proportions, not merely whether
it beats the received global model. FedPer can be closer than the average mix
(positive advantage, positive Pearson) and still have R² near 0.

### Last-upload advantage

Still computed, and still the x-axis of `boundary_logit.png`.

```
advantage = (1 - TV(π, π̂_upload)) - (1 - TV(π, π̂_global_received))
TV(π, π̂) = (1/2) Σ_k |π_k - π̂_k|
```

The probe is `test1`, shuffle off. FedKPer `infer` reads batches
`0 .. privacy_probe_batches-1` (default 8). The logged attack uses the next
`privacy_round_batches` batches (16). `last_advantage` is each client's last
participation, then the mean over clients.

**Means.** How much closer the upload is to π than the global model that
client just received. Near zero means this attack learns nothing beyond that
global model. It can be slightly negative. A method can zero this and still
leak π to a different attacker.

### Whispers of Data

**How.** The server records per-class accuracy of each upload on the public
probe. An LSTM with temporal attention, trained on virtual clients of the
victim's set size, maps that rounds-by-classes matrix to π (Ma et al.,
arXiv:2504.21436). This copy uses 32 virtual clients and four successive
local runs from the seed-0 global model. The victim's set size is given
(median client). The auxiliary images are the training pool. The code is
`fedseismic/privacy/whispers.py`.

**Means.** A reader of the upload trajectory. It recovers FedAvg. It does not
recover FedLC. It is not the paper's 1200 virtual clients.

### Secure aggregation on the figure

**How.** No protocol is run. The plotted R² is the score of assigning the
received global softmax to every client. The y value is the FedAvg fine-tune,
because that model can stay on the client. The same x is used on both rows:
Whispers has no per-client trajectory when the server only sees the sum.

**Means.** The mix guess available without a client upload. It is worse than
predicting the average fraction. If the fine-tuned weights are later probed,
the advantage comes back.

### DP-FedAvg

**How.** Updates are clipped and Gaussian noise is added to the averaged
delta after the upload has been scored. σ = 0.1 and 1.0 diverged and are
omitted. There is no moments accountant, so no ε is quoted.

**Means.** This noise does not change the upload the attacker reads.

### Presence

**How.** A class is present if its train mass is above `1e-3`. The score is
`π̂_upload,k - π̂_global,k`. Rows pool clients, rounds, and seeds, so they are
not independent samples. AUC ranks present against absent.

**Means.** Whether the server can tell which classes the client has.
Histogram R² can fall while this rises.

---

## Reconstruction

**How.** Geiping et al., NeurIPS 2020. One labeled image. The dummy starts at
`torch.randn` and is optimized to match the true one-step gradient in cosine
similarity, plus total variation. The label is known. FedPer drops the
classifier parameters from both gradients. This CNN has no BatchNorm. The
server stores a five-epoch delta, not this one step.

**Means.** Whether one gradient reveals the picture. PSNR is the literature
number and does not decide recognizability: the CIFAR car is identifiable at
13–15 dB, next to the blank-image floor. The comparison used here is SSIM
against the random start. A blank image is shown beside it. LPIPS is an
ImageNet embedding and is not the shared metric across CIFAR, MedMNIST, and
seismic. On a seismic slice the content score is per-class IoU of the facies
read from the recovered slice.

### Membership

**How.** Up to 64 of a client's train examples versus up to 64 of that
client's holdout. AUC near 0.5 is chance.

**Means.** Example membership, not the class mix. Appendix only.

---

## Other estimators

- **Softmax prior.** Mean softmax of the upload on the public probe. The CSV
  column `leakage` is `1 - TV` of this vector. It is not the plot x-axis.
- **Global softmax.** The same probe through the global model the client
  received.
- **Last-layer prior.** Softmax of last-layer row norms. FedPer's saved
  upload has the global head, so this number is not the private head.
- **Full-delta ridge.** Ridge from a sketch of `(upload - global)` to π,
  leave one client out, against `global_final`. Weaker than the softmax.
- **KL.** `Σ π log(π / π̂)` after smoothing. Lower is a stronger attack. The
  figure does not use it.

FedKPer's aggregation weight, in `infer` mode, is train accuracy times the
normalized entropy of the server's own π̂. That estimate is a
property-inference channel. `oracle` and `upload` modes are not on the figure.

---

## Sweep columns

| Column | Read it as |
|---|---|
| `local_mean` | Deployed local model on client holdouts. Plot y. |
| `global_score` | Aggregated model on the official test set. |
| `global_on_local` | Aggregated model on client holdouts. |
| `personalization_gap` | `local_mean - global_on_local`. Not the plot y. |
| `leakage` | Softmax `1 - TV` of the upload. Not the plot x. |
| class-mix R² | `r2_score` of predicted fractions. Plot x. |
| last-upload advantage | Upload leakage minus the received global. Older plot x. |

Code: `fedseismic/eval/metrics.py`, `fedseismic/privacy/metrics.py`,
`fedseismic/privacy/distribution.py`, `fedseismic/privacy/whispers.py`,
`fedseismic/privacy/boundary_r2.py`, `fedseismic/privacy/round_scores.py`.

---

## Prior floor and client bootstrap

The prior-only floor scores every client with the population mix, the mean of the clients' train shares in that seed. Presence AUC of that score is the floor. Leakage is the attack AUC minus the floor. For proportions, R² = 0 is the same floor, because `r2_score` compares against the mean share.

Intervals resample clients inside each seed, 1000 times, and take the 2.5 and 97.5 percentiles. A seed's statistic is recomputed on the resampled clients, then averaged across seeds. Classes and rounds of one client stay together.

## Bias slope

The bias change is the classifier bias of the upload minus the bias of the model that client received, averaged over the client's participations. One ordinary-least-squares slope pools clients and classes. The centered intercept is the intercept plus the slope divided by the number of classes. It is zero when the change is proportional to π − 1/K.

## Share bins

Absent is a train share at or below `1e-3`. The other bins are (0.001, 0.03], (0.03, 0.075], (0.075, 0.175], and a share of at least 0.25. The first three are centred on 0.01, 0.05, and 0.10. A share in (0.175, 0.25) is in none of them.

## Decaf

On one participation, a class is absent under the hard rule when no entry of its classifier-row change is positive. A client is called absent for that class only when this is true of every participation. The continuous score is the sum of the positive entries, averaged over participations. Higher means present. Proportions use a line from that score to the class share, fit on shadow clients, with hard-absent classes set to zero before normalizing.

## How much

The count of class k is n_k. The dataset size is n = Σ_k n_k. The proportion is π_k = n_k / n. A class is in the support when π_k > 1e-3. These scores are for an observer who sees one client's update (server-individual). The code is `fedseismic/privacy/howmuch.py`.

### Counts R²

**How.** Compare predicted counts with true counts over client-class pairs. The denominator is the residual of predicting, for each class, the mean count of that class across the clients in the seed. That mean is the population-count floor, so this R² is already the amount above the floor. It is not `r2_score` on the pooled vector: that baseline is one grand mean and mixes classes of different typical counts.

**Means.** Whether the update names how many examples of each class the client holds.

### Size R²

**How.** `r2_score` of predicted n against true n, one number per client. The baseline is the mean n in the seed.

**Means.** Whether the update names the size of the client's dataset. The floor is 0.

### Proportions given support

**How.** The truth on a client is π renormalized on its true support and zero elsewhere. The baseline inside the denominator is 1/|S| on that support and zero elsewhere, so predicting the uniform distribution on the true support scores 0.

The oracle attacker is told the support and renormalizes its non-negative count estimates on that support. The estimated attacker first marks a support of its own, with a logistic fit on shadow clients, and renormalizes on the support it marked. A wrong support lowers the score. Both floors are 0 by construction.

**Means.** The fraction among the classes the client has, separated from the question of which classes those are.

### Support

**How.** Presence AUC, and the true-positive rate at a false-positive rate of 10%. The prior-only floor scores every client with the population mix. Leakage is the attack number minus that floor.

**Means.** Which classes are present. Kept because a rare class can be the whole secret.

### Pooled mix R²

`r2_score` of the predicted proportions against π, classes pooled. It mixes support and proportions. It is kept for comparison with earlier tables and is labelled that way. It is not the how-much axis.
