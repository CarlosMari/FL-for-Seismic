# Findings for review

Question: on a federated classification task, can a client keep a model that
fits its own label mix without the upload telling the server that mix?

We have not gone to seismic. The evidence below is label shift on small
images. A 100-client, 100-round CIFAR-10 run of the same four methods is in
progress; the CIFAR numbers here are the finished 20-client, 40-round grid.

Metric definitions are in `metrics.md`. Short version: x is last-upload
leakage advantage (upload softmax versus the global model that client just
received, on a public probe disjoint from the one FedKPer uses to aggregate).
Y is mean accuracy of the deployed local model on each client's own holdout.
The official test accuracy is a third number and is the one to compare with
papers.

## Protocol

Dirichlet α=0.1, 10% participation, 5 local epochs, SGD lr 0.01, momentum 0.9.
Seeds 0, 1, 42. BloodMNIST: 20 clients, 8 classes. OrganCMNIST: 30 clients,
11 classes. CIFAR-10 grid below: 20 clients. Rounds: 40, except the rare-class
rehearsal (80 clients, 20 rounds). Model: the small CNN in this repo, not a
ResNet. FedKPer λ = min(λ_cap, 1 / teacher CE), KL of teacher softmax into the
student. FedLC trains cross-entropy on logits + log π and adds log π again
only when the client scores its own model. FedPer averages the trunk and
leaves the classifier on the client. The file on disk for FedPer is the
upload, with the global head put back. Aggregation mode for every FedKPer
number in the table is `infer`: the server estimates π̂ from the upload and
weights the average by that estimate's entropy times the client's reported
train accuracy. That estimate is itself a property-inference channel. If it
stopped tracking π, the aggregation weight would stop meaning what the method
wants it to mean.

## Threat model

Honest-but-curious server. No secure aggregation. The global model sent down
is the honest aggregate, not a crafted one. A malicious server that sends
weights chosen to amplify leakage is out of scope. The reason to study the
upload at all, rather than hide it under secure aggregation, is that the
server in this setting uses per-client updates: FedKPer's weights, and later
a seismic operator who may need to know which site an update came from.

## Dirichlet label shift

Means of three seeds. Last-upload advantage, then local holdout accuracy.

| Method | Blood advantage / local | OrganC advantage / local |
|---|---|---|
| FedAvg | 0.39 / 0.88 | 0.32 / 0.93 |
| FedProx | 0.38 / 0.88 | 0.32 / 0.93 |
| Ditto | 0.38 / 0.90 | 0.31 / 0.92 |
| FedKPer λ=0 | 0.33 / 0.90 | 0.32 / 0.93 |
| FedKPer λ=1 | 0.16 / 0.92 | 0.13 / 0.94 |
| FedKPer λ=10 | 0.14 / 0.92 | 0.12 / 0.94 |
| FedPer | 0.02 / 0.93 | 0.03 / 0.92 |
| FedLC | 0.01 / 0.93 | 0.03 / 0.94 |
| FedKPer λ=10 + same logit loss | 0.00 / 0.94 | 0.00 / 0.94 |

Per-seed ranges for the FedKPer drop do not overlap FedAvg (Blood λ=10 is
0.09, 0.18, 0.18 against FedAvg 0.35–0.39). Averaging every participation
instead of the last one raises every method and keeps the same order (Blood
FedAvg 0.44 to FedKPer λ=10 0.21).

A final local fine-tune of the saved FedAvg global reaches local accuracy
0.91 Blood / 0.95 OrganC. The server still only received the FedAvg upload.
If that fine-tuned model is probed, advantage comes back (0.33 Blood / 0.28
OrganC).

Putting the true log π on the global model at inference closes 87–101% of the
local-accuracy gap. A presence-only correction (absent classes down, the rest
uniform) recovers about half on Blood and about 60% on OrganC. Flooring absent
classes at 1% and keeping the proportions matches the full correction. So the
accuracy gap on this split is the mix, not only the support.

On the client holdout, the local model is ahead on every Blood class. The gap
is largest where the global model is weak: classes 2, 3, and 4 are about
0.89–0.91 local versus 0.24–0.27 global under FedAvg. FedKPer λ=10 lifts the
global model on classes 0, 2, and 4; class 3 stays local (0.95 vs 0.33).
OrganC is the same pattern with smaller gaps. Macro-F1 on classes present
locally: Blood FedAvg 0.73 vs 0.55 global, FedKPer 0.81 vs 0.66. OrganC
0.83 vs 0.75 and 0.87 vs 0.82.

Worst 10% of Blood clients: local 0.57 to 0.71 from FedAvg to FedKPer λ=10.
The global model on those same clients only goes 0.46 to 0.55.

Round-pooled presence AUC on the Dirichlet softmax residual does not fall with
λ the way the histogram does (Blood about 0.77 to 0.84 when rounds are
pooled). Under FedLC it falls to roughly 0.55–0.61, which is close to chance
on that same residual. Those rows reuse clients across rounds.

## When a class is actually missing

80 clients, rarest two classes held out of half the clients and kept at 1–5%
in the rest. About 240 negative clients per rare class (still pooled over
rounds).

FedAvg to FedKPer λ=10, last-upload advantage: Blood 0.18 to 0.10, OrganC
0.15 to 0.07. Presence AUC goes the other way: Blood 0.84 to 0.88, OrganC
0.86 to 0.91. TPR at 1% FPR: Blood 0.41 to 0.73, OrganC 0.35 to 0.68. Local
accuracy does not beat the global model (Blood 0.68 / 0.70 vs about 0.72;
OrganC 0.74 / 0.77 vs 0.79 / 0.81).

Histogram leakage can drop while presence gets easier to read, and on this
split personalization is not buying accuracy.

## Images

One labeled image per client, seed 0, dummy from noise, cosine match to the
true one-step gradient plus total variation. Label known, so this is an upper
bound on that attack. It is not the five-epoch update the server stores.

A blank image already scores about 10 dB Blood, 16 dB OrganC, 12 dB CIFAR,
because the images are smooth. FedAvg lands on that floor (Blood 11, OrganC
17, CIFAR 13). FedPer on CIFAR reaches 18 dB. The reconstruction is a blurry
car, not a pixel copy. FedKPer does not hide the image relative to FedAvg.

Membership AUC of local models on each client's own holdout is about 0.49–0.53.

## CIFAR-10, and why 0.89 is not the paper number

20 clients, 40 rounds, same CNN and same four methods.

| Method | Last-upload advantage | Local holdout | Global on local holdouts | Official test |
|---|---|---|---|---|
| FedAvg | 0.49 | 0.87 | 0.33 | 0.33 |
| FedKPer λ=10 | 0.25 | 0.88 | 0.52 | 0.52 |
| FedPer | 0.04 | 0.88 | 0.63 | 0.62 |
| FedLC | 0.05 | 0.89 | 0.67 | 0.67 |

Seed 0 FedAvg: train and holdout indices are disjoint. Mean majority-class
rate on the client train sets is 0.62, and the local model is about 0.87, so
it is above the prior. Official test for that seed is 0.43. The three FedAvg
official-test seeds are 0.12, 0.43, 0.44.

The 0.89 figure is holdout accuracy under a 20-client α=0.1 split. Papers
usually quote official test accuracy, often with 100 clients. We are rerunning
at 100 clients and 100 rounds. The architecture is still this CNN. Seed 0 of
that run finished at official test 0.52. Per-round accuracy was not saved, so
that number is not a stability check.

## Other attackers on the saved checkpoints

Same probe window, but the baseline global is `global_final.pt`, not the
global the client received. Advantage is still leakage of the attack minus
leakage of that global model's softmax. Means of three seeds
(`results/attack_suite.csv`).

| Method | Blood softmax / offset / row | OrganC | CIFAR |
|---|---|---|---|
| FedAvg | 0.37 / 0.27 / 0.02 | 0.31 / 0.34 / 0.00 | 0.52 / 0.49 / 0.06 |
| FedKPer λ=10 | 0.15 / 0.13 / −0.01 | 0.12 / 0.12 / −0.01 | 0.26 / 0.28 / 0.02 |
| FedPer | 0.02 / 0.01 / 0.00 | 0.02 / 0.01 / −0.01 | 0.05 / 0.04 / 0.00 |
| FedLC | 0.01 / −0.07 / −0.01 | 0.03 / −0.06 / −0.01 | 0.05 / 0.02 / 0.01 |

Logit offset is not sharper than mean softmax here. Classifier row-norms, turned
into a softmax, do not beat the global model. FedPer's row norm is 0 because
the file on disk has the global head. That does not say the trunk is clean.
The shadow regressor on the full delta is not run yet.

On Dirichlet, present and absent classes have almost the same row norm under
FedLC (Blood 0.14 vs 0.14). "Absent" at a 1e-3 threshold is often one example.
The rehearsal, where absence is real, does not include FedLC. There FedAvg's
absent rows are only slightly smaller (Blood 0.15 vs 0.17, OrganC 0.13 vs 0.15).

Softmax of the last-upload bias delta, compared with the global the client
actually received, has advantage about 0.05 for Blood FedAvg and about 0 or
negative for FedKPer, FedPer, and FedLC (`results/bias_advantage.csv`).

Per-class d′ of the softmax residual does not fall as FedKPer λ rises. Blood
Dirichlet: FedAvg 1.03, λ=0 1.07, λ=1 1.32, λ=10 1.26. FedLC is 0.32. On the
rehearsal, Blood goes 1.04 to 1.15 and OrganC 1.14 to 1.06 (`results/residual_dprime.csv`).
Those rows are participations, not independent clients.

## What we think this supports

On pure label shift, matching local accuracy does not require an upload whose
public softmax tracks π. FedPer and FedLC are near zero on that attack, and
also on logit offset and classifier row-norms. That is not yet a privacy
claim: FedPer's head never leaves, and the full-delta regressor is still open.
FedKPer cuts the softmax advantage by about half and also lifts the global
model. Ditto does not change the upload. Fine-tuning at the end personalizes,
and the personalized weights still carry the mix if anyone probes them.

The failure mode we trust more for seismic is the rehearsal: when absence is
real, FedKPer's softmax residual stays informative about presence, and the
local model is not more accurate than the global one. d′ does not show a clean
rise with λ.

## Asks

1. Is last-upload softmax advantage the right x-axis once FedLC can zero it
   by adding log π only at local scoring? The features can still carry the mix.
2. For seismic, should the claim be about presence of rare classes rather than
   histogram TV? The rehearsal is the result that worries us.
3. Is the one-step inversion enough to say the image is not recovered, given
   FedPer's blurry car, or is the next measurement the five-epoch delta?
4. Before any paper comparison on CIFAR, is 100 clients, 100 rounds, and this
   CNN the right bar, or do we need the architecture those tables use?
5. Which personalization method is worth adding before seismic: FedRoD,
   a local head on early features, or something aimed at feature shift rather
   than another label-shift correction?
