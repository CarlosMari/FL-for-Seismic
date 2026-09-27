# Findings for review

Question: on a federated classification task, can a client keep a model that
fits its own label mix without the upload telling the server that mix?

We want a decision on what to run next. The classification evidence is below.
Seismic has not been started. Metric definitions are in `metrics.md`. The
figure to look at is `results/figures/boundary_r2.png`.

## What we think is supported

On Dirichlet label shift, the local-accuracy gap is the prior. Putting the
client's true log π on the shared model at inference closes 87–101% of that
gap. The prior can stay on the client.

The public softmax of the upload is a weak attacker. FedPer, FedLC, logit
adjustment, and FedRoD drive its class-mix R² to about zero. That is not a
privacy claim about the weights. FedPer's classifier never leaves the client.
FedLC and logit adjustment are trained so that the softmax, read without the
prior, is a bad estimate of π.

A stronger reader of the same uploads, Whispers of Data, still recovers
FedAvg and FedBABU. It does not recover FedLC or logit adjustment (R²
negative, Pearson about 0). FedPer is in between: closer than the average
mix, with a positive correlation and an R² near 0 on Blood and OrganC, and
0.21 on CIFAR-10. A negative R² is worse than predicting the average
fraction. It is not a reversed correlation.

FedBABU does not change this. The frozen classifier is fit once at the end
and is not uploaded, but the body that is uploaded still reads like FedAvg,
and local accuracy is lower. FedRoD does change the upload: softmax R² is
near 0 and Whispers R² is negative. Local accuracy drops with it, and on
macro-F1 the deployed FedRoD model does not beat the shared model on Blood
or OrganC.

DP-FedAvg, as implemented, cannot hide this upload. The noise is added to
the average after the upload is scored. σ = 0.01 leaves the class mix
readable and cuts CIFAR-10 local accuracy to 0.78. We do not quote ε. There
is no moments accountant, and σ = 0.1 and 1.0 diverged.

Secure aggregation is not a run. With every selected client finishing the
round, the server sees the same average as FedAvg and has no per-client
upload. The only mix it can assign to a client is the shared model's
softmax. That guess has class-mix R² −0.30 / −0.08 / −0.26 on Blood,
OrganC, and CIFAR-10. The client's own model can still be a FedAvg
fine-tune. If those fine-tuned weights are probed, the advantage comes back
(about 0.33 Blood / 0.28 OrganC).

Image recovery is a separate axis. Hiding the mix does not hide the picture
on the one CIFAR car we inverted. The Blood cell stays near the random start.

## How to read the figure

`results/figures/boundary_r2.png`. Dirichlet α = 0.1, mean of seeds 0, 1, 42.
No error bars and no lines. Y is mean local holdout accuracy of the deployed
model. Higher is better.

X is class-mix R²: predicted class fraction against the true train fraction,
classes pooled. 1 is a perfect mix. 0 is the score of predicting every
client's average fraction. Negative is worse than that average. Pearson is
reported beside it and is not what R² squares.

The top row is the last-upload softmax. The bottom row is Whispers. The
green star is the secure-aggregation placement above, on both rows, because
Whispers has no per-client trajectory either. The black triangle is FedAvg
plus a local fine-tune, drawn at FedAvg's measured R², because that upload
was visible.

## Protocol

Dirichlet α = 0.1, 10% participation, 5 local epochs, SGD, learning rate
0.01, momentum 0.9. BloodMNIST: 20 clients, 8 classes. OrganCMNIST: 30
clients, 11 classes. CIFAR-10 grid: 20 clients, 40 rounds, unless a sentence
says 100 clients and 100 rounds. Model: the small CNN in this repo
(`FedAvgNetCIFAR`: two convolutions and a linear layer; MedMNIST uses the
same family). It is not a ResNet. The CNN has no BatchNorm.

FedKPer numbers are aggregation mode `infer`: the server estimates π̂ from
the upload and weights the average by that estimate's entropy times the
client's train accuracy. That estimate is itself a property-inference
channel. FedLC trains cross-entropy on logits + log π and adds log π again
only when the client scores its own model. FedPer averages the trunk and
leaves the classifier on the client. The saved upload has the global head
put back; the deployed head is a separate file. FedBABU freezes the
classifier at the shared initialization, averages the body, and fits the
classifier once after the last round. FedRoD trains a shared generic head
and a personal residual on the same features. The residual starts at zero,
stays on the client, and the loss is cross-entropy on the summed logits
plus log π. The deployed model is the sum of the two heads. Both new methods
use the same five local epochs as the rest of the grid.

## Threat model

Honest-but-curious server. No secure aggregation in the runs. The global
model sent down is the honest aggregate. A malicious server that sends
weights chosen to amplify leakage is out of scope. The reason to study the
upload is that some methods use the per-client update: FedKPer's weights,
and later a seismic operator who may need to know which site an update came
from.

## Class mix

Means of three seeds. Softmax R² is the top row of the figure. Whispers R²
is the bottom row. Local accuracy is the deployed model. For FedAvg,
FedKPer, FedPer, and FedLC the two rows come from separate runs and the
local accuracies agree within about 0.01.

| Method | Blood softmax / Whispers / local | OrganC | CIFAR-10 |
|---|---|---|---|
| FedAvg | 0.65 / 0.35 / 0.88 | 0.56 / 0.36 / 0.93 | 0.81 / 0.69 / 0.87 |
| FedKPer λ=10 | 0.33 / 0.05 / 0.92 | 0.26 / 0.17 / 0.94 | 0.52 / 0.52 / 0.88 |
| FedPer | 0.01 / 0.04 / 0.93 | 0.05 / −0.01 / 0.92 | 0.10 / 0.21 / 0.88 |
| FedLC | 0.00 / −0.35 / 0.93 | 0.08 / −0.19 / 0.94 | 0.11 / −0.47 / 0.89 |
| FedKPer λ=10 + logit | −0.01 / −0.39 / 0.94 | −0.01 / −0.44 / 0.94 | −0.01 / −0.36 / 0.89 |
| FedBABU | 0.64 / 0.38 / 0.89 | 0.58 / 0.38 / 0.90 | 0.84 / 0.81 / 0.84 |
| FedRoD | 0.01 / −0.38 / 0.81 | 0.05 / −0.16 / 0.88 | 0.08 / −0.35 / 0.76 |
| DP-FedAvg σ=0.01 | 0.67 / 0.34 / 0.90 | 0.62 / 0.38 / 0.93 | 0.90 / 0.76 / 0.78 |
| SecAgg placement | −0.30 / −0.30 / 0.91 | −0.08 / −0.08 / 0.95 | −0.26 / −0.26 / 0.88 |

FedProx and Ditto sit on FedAvg. FedKPer λ=0 does too. λ=1 is between λ=0
and λ=10. Those four are on the figure for Blood and OrganC and were not
rerun on CIFAR-10. CIFAR FedKPer's Whispers mean is pulled by seed 1
(R² 0.78); seeds 0 and 42 are 0.30 and 0.47.

Whispers Pearson, where we have it: FedAvg about +0.65 to +0.84. FedPer
about +0.45 to +0.54. FedLC and logit adjustment about 0 to +0.15. FedBABU
matches FedAvg (+0.66 to +0.91). FedRoD is about +0.14 to +0.17. The FedPer
pattern is a real correlation with a slope too flat for a high R². The FedLC
pattern has nothing to flip.

Whispers here is a reduced copy of Ma et al. (arXiv:2504.21436): 32 virtual
clients, four successive local runs, the victim's set size given as the
median client (Blood 341, OrganC 294, CIFAR 1670). The auxiliary data is the
training pool, which is optimistic for the attacker. It is not their 1200
virtual clients.

The older x-axis, last-upload total-variation advantage, still orders the
methods the same way on the softmax. Blood FedAvg 0.39 to FedKPer λ=10 0.14,
FedPer 0.02, FedLC 0.01. It is the wrong summary once a method can be
"closer than the average mix" without a readable histogram. That is why the
figure uses R².

A ridge on the full weight delta, against `global_final`, is a weaker
histogram reader than the softmax. Blood advantage about 0.10 for FedAvg,
0.07 for FedKPer λ=10, about 0 for FedPer and FedLC. It does not clear the
weights.

## Utility

Macro-F1 is computed on classes present in the client's train mix (π > 1e-3),
scored on the holdout, then averaged over clients. Mean accuracy treats every
client equally and rewards majority classes. FedLC is scored with log π at
inference. FedPer, FedBABU, and FedRoD are scored from the saved personal
head.

| Method | Blood local / global F1 | OrganC | CIFAR-10 |
|---|---|---|---|
| FedAvg | 0.73 / 0.53 | 0.83 / 0.76 | 0.60 / 0.29 |
| FedKPer λ=10 | 0.80 / 0.65 | 0.85 / 0.81 | 0.65 / 0.46 |
| FedPer | 0.81 / 0.73 | 0.83 / 0.81 | 0.65 / 0.55 |
| FedLC | 0.84 / 0.75 | 0.86 / 0.84 | 0.66 / 0.59 |
| FedKPer λ=10 + logit | 0.82 / 0.76 | 0.86 / 0.81 | — |
| FedBABU | 0.76 / 0.54 | 0.81 / 0.73 | 0.56 / 0.27 |
| FedRoD | 0.77 / 0.77 | 0.83 / 0.85 | 0.62 / 0.60 |

On Blood the accuracy gap sits on classes the shared model misses (classes
2, 3, and 4 under FedAvg). One Blood FedPer seed is an accuracy win and a
macro-F1 tie. CIFAR-10 FedAvg seed 1 is a chance global model (official test
0.12, global macro-F1 0.10) and pulls that column down.

Official test accuracy, 20 clients and 40 rounds: FedAvg 0.33, FedKPer 0.52,
FedPer 0.62, FedLC 0.67. The 100-client, 100-round run, official test only:
FedAvg 0.58, FedKPer 0.59, FedPer 0.60, FedLC 0.66. Leakage was not scored
there. Do not compare either number to a paper unless client count, α,
participation, local epochs, rounds, and architecture all match.

## Presence, when absence is real

80 clients, rarest two classes held out of half the clients and kept at 1–5%
in the rest. Histogram advantage falls from FedAvg to FedKPer while
rare-class presence AUC rises (Blood 0.84 to 0.88, OrganC 0.86 to 0.91).
FedPer and FedLC bring that AUC down (Blood 0.77 and 0.66, OrganC 0.76 and
0.60). On this split the local model does not beat the shared model, except
Blood FedLC by about 0.01. At 20 clients the local sets are larger and the
accuracy gap shrinks to a tie. FedKPer still does not win locally. Rows are
participations, so the AUC is not an independent-client sample. A
client-cluster bootstrap on the 80-client run leaves the FedKPer intervals
nearly touching FedAvg.

## Images

The attack is Geiping et al., NeurIPS 2020: cosine similarity to the true
one-step gradient, plus total variation, dummy from `torch.randn`, label
known. It is a baseline, not current image SOTA. Current attacks add a
generative prior or BatchNorm statistics. This CNN has no BatchNorm, and a
natural-image prior is the wrong attacker for MedMNIST or a seismic slice
unless it is trained on public data from that domain. The server stores a
five-epoch delta, not this one step.

On client 0's CIFAR training image, seed 0, 4000 steps, all four saved
uploads recover a recognizable car. PSNR is 13–15 dB against a blank-image
floor of 12.8 dB, so PSNR calls it a miss. SSIM against the random start
(0.20) is FedAvg 0.62, FedKPer 0.71, FedPer 0.71, FedLC 0.77. The same
attack on one Blood cell stays near the start (SSIM 0.22–0.35 versus 0.20).
CIFAR label 1 in this pair is a car.

For a later seismic slice, the content metric should be per-class IoU of the
facies read from the recovered slice. Pixel PSNR and SSIM do not answer
that. LPIPS is an ImageNet embedding and is the wrong shared metric across
CIFAR, MedMNIST, and seismic.

Membership of local models on each client's holdout is about 0.49–0.53.
That stays in the appendix.

## What we are not claiming

FedPer and FedLC uploads have not been shown to hide π from every weight
attack. The full-delta ridge is weaker than the softmax, and Whispers does
not make FedPer's histogram readable. A better attacker could still find it.

The FedRoD accuracy drop is this recipe: five local epochs, a zero residual,
balanced softmax on the summed logits. It is not a sweep of the paper's
personal-head schedule.

The secure-aggregation star is the R² of a guess the server can make without
a client upload. It is not an implementation of Bonawitz et al.

No ε. Diverged DP runs are omitted. Seismic is not started.

## Questions

1. Is class-mix R² the right x-axis for the claim, given FedPer can have a
   positive Whispers advantage and a Pearson around +0.5 while R² stays near 0?
2. Is this reduced Whispers attack (32 virtual clients, set size given,
   training pool as auxiliary data) strong enough to say FedLC's mix is not
   recovered, or should the next attacker read the saved weights or the
   five-epoch delta?
3. FedRoD kills both readers and loses the accuracy. Should we treat that
   cost as real, or retrain with a longer personal-head phase before using
   the point?
4. Should the y-axis stay mean local accuracy, or move to macro-F1 on the
   classes the client actually has? One Blood FedPer seed is an accuracy win
   and a macro-F1 tie.
5. Is the secure-aggregation star, placed at the measured global-softmax R²
   without running a protocol, acceptable on the figure?
6. For reconstruction, is SSIM against the random start the shared image
   metric, with facies IoU reserved for F3 and Parihaka? Is Geiping et al.
   2020 enough as the attacker until the model has BatchNorm or a domain prior?
7. What is the next run: more FedRoD local steps, an attack on the five-epoch
   delta, or the seismic UNet (FedAvg against FedBN against a GroupNorm
   control, local head, facies IoU, contiguous blocks rather than Dirichlet)?
