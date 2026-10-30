> **RESEARCH AUDIT CORRECTION (30 Oct 2026):** Earlier entries contain unsupported
> assertions about the rotation, loss functions, and matched-size parity. The
> rotation/buffer comparison is confounded, not proof that either caused the
> 0.14 difference; the activation probes cannot establish orientation has no
> effect. The `fedpersurvey` trainer uses masked cross-entropy, not FocalDice.
> The completed matched-size Parihaka results have opposite signs across seeds;
> claims about consistent harm, conclusive non-parity, or ruling out a data-volume
> artifact are unsupported. See the corrected entries below.

> **CRITICAL PARITY AUDIT (30 Oct 2026):** All six two-survey parity runs
> used a defective `aggregate_survey_heads`: the 4-D convolutional classifier
> weight is treated as a bias and each class row is averaged to a scalar after
> every round, including in the one-client solo baseline. All six saved
> `global_final.pt` heads confirm zero across-feature standard deviation per row.
> The reported mIoUs are real scores from the saved local models, but do not
> establish the intended FL-vs-solo comparison. Do not use the paired deltas
> as a definitive result until the implementation is fixed and tested and the
> affected arms are rerun. The advisor confirmed that the currently running
> FedHCA2-style arm follows a different aggregation path; the queued
> fedpersurvey and solo baselines require the corrected code.

> **SUBSET METADATA CORRECTION (30 Oct 2026):** `labels/pi.json` in the matched
> runs was computed from the pre-subset pool, not the 350 sections actually
> used. Direct inspection of the F3 label cube shows zero class-5 pixels in
> training, but 534,099 class-5 pixels in the held-out test. Thus the earlier
> statement that class 5 was present in F3 training, based on `pi.json`, was
> wrong. This mismatch is also a reason the solo scores can be low.

> **FEDHCA2-STYLE HEAD CHECK (30 Oct 2026):** The running three-survey arm
> uses the generic server aggregation path and restores each client's own head
> from its personal state; it does not call `aggregate_survey_heads`. Inspection
> of the current personal checkpoints found nonzero channel-wise standard
> deviations in all active head rows (approximately 0.10–0.23), unlike the
> collapsed fedpersurvey aggregate. Regression tests also check its server path
> and personal-head restoration. This checks the specific head-collapse issue;
> it is not proof that the whole FedHCA2-style implementation is correct.

> **NAMING NOTE (Oct 2026) — read before citing any "FedLC" row below.**
> The method labelled **"FedLC"** throughout these notes is NOT FedLC (Zhang et al., ICML 2022).
> It is **logit-adjusted local training (LA)**: cross-entropy on `z + tau*log pi` (the class
> log-prior), with `log pi` re-applied at local inference. This is the logit-adjustment /
> balanced-softmax form (Menon et al., 2021; Ren et al., 2020, tau=1). FedLC instead subtracts a
> class-count MARGIN `tau*n_c^{-1/4}` (its Eq. 7), leaves the true class out of the denominator,
> and predicts with raw logits. We do NOT use FedLC and do not claim to reproduce it.
> - Thesis name: **LA** (logit-adjusted local training). Cite Menon 2021 / Ren 2020 as the source,
>   Zhang 2022 (FedLC) as related work, Chen & Chao 2022 (FedRoD) as the closest federated form.
> - Run/table identifiers still say `fedlc`; they mean LA. Kept as-is to match saved runs.
> - LA (logit_adjust) is NOT used in the seismic experiments (logit_adjust=False there).

# Findings

Question: can a client keep a model that fits its own mix of classes without the update it sends telling the server that mix?

Plain answer: not with the methods we have trained. Ordinary training leaks the mix through two separate clues in the last layer. Keeping the client's class correction on the client hides one clue and leaves the other wide open. A defense that tries to close both only reverses the first clue, and an attacker who notices the reversal can still read it. Accuracy gets worse at the same time.

Seismic has not been started. Metric definitions are in `metrics.md`. The older method comparison is `results/figures/boundary_r2_nodp.png`. The result that decides the next step is `results/tables/head_channels.csv`.

## What is supported

On Dirichlet label shift, the accuracy gap between a client's model and the shared model is mostly the client's class prior. That prior can stay on the client. It does not have to be uploaded.

The public softmax of an upload is a weak reader. FedPer, FedLC, logit adjustment, and FedRoD drive its class-mix R² to about zero. That happens because those methods are built so the softmax, read without the prior, is a bad estimate of the mix. It is not evidence that the weights hide the mix.

There are two clues in the classifier, and they are not the same clue.

- The bias change, upload minus the model the client just received, moves with the class count under ordinary training. Averaged over each client's four participations, Blood FedAvg is readable: Pearson +0.68, presence AUC 0.79, leave-one-out R² +0.45. FedLC, which adds log π only on the client, sits at chance on this clue (Pearson −0.20, AUC 0.53, leave-one-out R² +0.00).
- The size of each class's weight change does the opposite. FedLC presence AUC is 0.99, against 0.73 for FedAvg. Logit adjustment hides the class proportions and makes presence easier to detect. An absent class has π near 0, so log(π + ε) is very negative and the model assigns that class almost no probability. Those classes then get almost no gradient, and their classifier rows barely move. Under ordinary training, an absent class is still pushed down whenever the model predicts it, so its row moves too. Logit adjustment switches absent classes off, and the server can see which rows did not move. The exact fraction stays weakly recovered (leave-one-out R² +0.08). FedAvg is readable on this clue as well (Pearson +0.57, AUC 0.73, leave-one-out R² +0.30), so this is not a failed attack being asked to judge FedLC.

Resampling the 20 clients (1000 draws, 2.5–97.5%) keeps that gap. FedLC row-size presence AUC is 0.99, 0.99, and 0.97 on the three seeds, and the lowest bootstrap bound is 0.94. FedAvg's row-size AUC is 0.71, 0.68, and 0.80, and the highest bootstrap bound is 0.87. Twenty clients is too few negatives to quote a 1% false-alarm rate. The 10% rate is what these runs can support.

The same row size still separates an absent class from a small share, which is the fixed-ratio game. AUC of "absent" against "share at least this much," from the row size:

| share | FedAvg | FedLC |
|---|---|---|
| 0.01 | 0.76 | 0.99 |
| 0.05 | 0.83 | 0.98 |
| 0.10 | 0.86 | 0.98 |
| 0.25 | 0.90 | 0.97 |

The headline stays at 0.25. The small-share end is the one that matters for a rare facies, and FedLC does not get harder there. FedAvg does. The bias clue on FedLC stays at chance at every one of these cuts.

A combined defense was trained on the same Blood split: inverse-frequency loss, eight public images of every class held out of the pool before the split, and log π kept on the client. It fails, and it is not being tuned further. The bias score runs backwards (Pearson −0.67, AUC 0.10). Flipping it gives presence AUC about 0.90, and a line fit on the other clients still recovers leave-one-out R² +0.39. The row-size clue drops from 0.99 to 0.62. Local accuracy drops from 0.93 to 0.87. The likely cause of the reversed bias is that a class the client does not have, and which appears only in those eight public images, is fit quickly and looks different from a class the client actually holds. The server can separate "public only" from "real." Report that as the negative result.

## Smoothing does not close both clues

The training prior can be pulled toward uniform, π′ = (1−λ)π + λ/K. λ = 0 is FedLC. λ = 1 is a constant prior, which is FedAvg. Blood, 20 clients, 40 rounds, five seeds (0, 1, 2, 3, 42), no public pad. The table is `results/tables/prior_smooth.csv`. Means:

| λ | Row size: Pearson / AUC | Bias: Pearson / AUC | Best presence AUC | Learned absent-vs-0.25 accuracy | Local accuracy |
|---|---|---|---|---:|---:|
| 0 (FedLC) | +0.44 / 0.98 | −0.09 / 0.50 | 0.98 (row) | 0.88 | 0.932 |
| 0.02 | +0.45 / 0.89 | +0.14 / 0.76 | 0.89 (row) | 0.89 | 0.935 |
| 0.1 | +0.48 / 0.81 | +0.38 / 0.84 | 0.84 (bias) | 0.88 | 0.932 |
| 0.3 | +0.54 / 0.75 | +0.57 / 0.85 | 0.85 (bias) | 0.89 | 0.930 |
| 1 (FedAvg) | +0.66 / 0.70 | +0.74 / 0.82 | 0.82 (bias) | 0.92 | 0.905 |

There is no setting where the best clue falls to chance. The lowest best-clue presence AUC on this curve is 0.82. As λ grows, the row channel weakens and the bias channel reopens. The leak moves. Local accuracy stays near 0.93 until λ = 1, where it drops to 0.91, so the accurate settings are exactly the ones that still leak.

The same two endpoints, five seeds, on OrganC (30 clients) and CIFAR-10 (20 clients):

| Dataset | FedLC row AUC | FedLC bias AUC | FedAvg row AUC | FedAvg bias AUC | FedLC local | FedAvg local |
|---|---:|---:|---:|---:|---:|---:|
| Blood | 0.98 | 0.50 | 0.70 | 0.82 | 0.932 | 0.905 |
| OrganC | 1.00 | 0.47 | 0.73 | 0.82 | 0.935 | 0.917 |
| CIFAR-10 | 0.99 | 0.53 | 0.72 | 0.87 | 0.886 | 0.852 |

FedLC bias stays at chance on all three. The row-size clue does not. The learned attacker tells an absent class from a share of at least 0.25 at accuracy 0.88 on Blood, 0.94 on OrganC, and 0.96 on CIFAR-10. Chance is 0.50. The full λ curve is Blood only. OrganC and CIFAR-10 are the two endpoints.

FedBABU does not help. The classifier is frozen and fit once at the end, but the body that is uploaded still reads like ordinary training, and accuracy is lower. FedPer never uploads the classifier, so both clues above are exactly zero on the saved head. That does not show the shared body is safe: a linear probe of those bodies could not read ordinary training either, so a blank FedPer result says nothing.

FedRoD's low accuracy on the figure is the training recipe. log π was added to the sum of the shared head and the personal head, so the personal head learned to cancel the prior. Adding log π only at scoring time lifts mean local accuracy from 0.809 to 0.930 on Blood, 0.884 to 0.938 on OrganC, and 0.762 to 0.884 on CIFAR-10. The upload is still the shared head. That retrain, with the paper's split of the two losses, has not been run. It should not be expected to hide the mix better than FedLC.

Server-side DP does not hide the upload. The noise is added to the average after the upload is scored. We do not quote ε. Those points are off the figure.

Secure aggregation was not run. A round has two clients on Blood and CIFAR-10 and three on OrganC. If each upload revealed the mix exactly, assigning every member the average of that round's true mixes would score about R² +0.56 on Blood and +0.46 on CIFAR-10. The old star, which used the shared model's softmax (R² about −0.30 / −0.08 / −0.26), is a worse guess than predicting the population mix, and it is off the figure. The weight sums themselves were not attacked: only the last round's uploads were saved.

Image recovery is a separate question, and one CIFAR car plus one Blood cell is not enough to carry it. It is not needed for the class-mix claim.

## The channel measurement

BloodMNIST, Dirichlet α = 0.1, 20 clients, 40 rounds, 10% participation, 5 local epochs. Seeds 0, 1, 42. Eight images per class were held out of the training pool before the split, so FedAvg, FedLC, and the defense see the same clients. Only the defense trains on those images. Each score is the mean of a client's participations. The reference is the model that client received, not the model at the end of training. Leave-one-out R² fits one line on the other clients, so a flipped sign still counts as readable.

| Method | Bias: Pearson / AUC / leave-one-out R² | Row size: Pearson / AUC / leave-one-out R² | Local accuracy |
|---|---|---|---|
| FedAvg | +0.68 / 0.79 / +0.45 | +0.57 / 0.73 / +0.30 | 0.911 |
| FedLC | −0.20 / 0.53 / +0.00 | +0.37 / 0.99 / +0.08 | 0.927 |
| Defense | −0.67 / 0.10 / +0.39 | +0.26 / 0.62 / +0.03 | 0.867 |

These local accuracies are not the ones on the older figure. That figure did not hold out a public pad.

## Learned attacker

The attacker above is hand-built: one clue, one score. The learned attacker sees both clues at once. It is fit on 48 shadow clients that use the victim's own loss for five local epochs, starting from that seed's final global model. It never sees a real client during the fit. Real clients are then scored from the classifier change they uploaded. The table is `results/tables/meta_attacker.csv`. Means of the same three Blood seeds:

| Method | Best hand-built presence AUC | Learned presence AUC | True-positive rate at 10% false alarms | Absent vs high-share accuracy |
|---|---:|---:|---:|---:|
| FedAvg | 0.79 (bias) | 0.72 | 0.45 | 0.93 |
| FedLC | 0.99 (row size) | 0.90 | 0.71 | 0.87 |
| Defense | 0.90 (bias, sign flipped) | 0.79 | 0.52 | 0.85 |

Chance on the last column is 0.50. Every method is above it. FedLC is the easiest presence detector: at a 10% false-alarm rate the learned attacker still catches about 71% of the classes that are actually present. The defense does not survive this. Telling an absent class from one that is at least a quarter of the client's data succeeds 85% of the time.

The learned mix is about as good as the best single clue, not better. On FedAvg the bias alone already has leave-one-out R² +0.45. On FedLC the exact fraction stays weak (learned leave-one-out R² +0.06). What is recovered is which classes are there.

That table starts every shadow from the final global model, while the real update was taken against the model received that round. It is a lower bound. The curve below removes the mismatch: each shadow starts from a received model saved during the run, drawn across the 40 rounds. That is an LDIA-style attacker (Gu and Bai, Journal of Information Security and Applications, 2023). It still does not beat the best single clue. The number to report is the best attacker overall.

## How to read the older figure

`results/figures/boundary_r2_nodp.png`. Same protocol, no public pad. Y is mean local holdout accuracy of the deployed model. Higher is better. X is class-mix R² of a predicted mix against the true mix, classes pooled.

R² = 0 is the uniform mix, one over the number of classes. It is not the average client's mix. The population mix scores slightly above 0 (about +0.02). A negative R² means the attacker did worse than guessing the uniform mix. It does not mean the mix is hidden, and methods should not be ranked by how negative they go. Pearson is reported separately. It is not what R² squares.

The top row is the last-upload softmax. The bottom row is Whispers. Secure aggregation and server-side DP are not on this figure.

A leave-one-client line does not rescue the softmax of FedLC or FedPer. On Blood seed 0 the FedAvg softmax moves from R² +0.67 to +0.68. FedLC stays at about +0.00 and FedPer at about +0.00.

Whispers of FedLC was trained with FedLC's own loss, so the negative R² is not a train/test mismatch. FedPer's Whispers Pearson is about +0.45 to +0.54, so a line fit could recover on the order of 0.2–0.3. The predicted vectors were not saved, so that fit was not run. Whispers is 32 virtual clients and four local steps, with the training pool as auxiliary data. That is optimistic for the attacker, and it is not a reason to say FedLC's mix is unrecovered. The row-size result above is the one that recovers it.

| Method | Blood softmax / Whispers / local | OrganC | CIFAR-10 |
|---|---|---|---|
| FedAvg | 0.65 / 0.35 / 0.88 | 0.56 / 0.36 / 0.93 | 0.81 / 0.69 / 0.87 |
| FedKPer λ=10 | 0.33 / 0.05 / 0.92 | 0.26 / 0.17 / 0.94 | 0.52 / 0.52 / 0.88 |
| FedPer | 0.01 / 0.04 / 0.93 | 0.05 / −0.01 / 0.92 | 0.10 / 0.21 / 0.88 |
| FedLC | 0.00 / −0.35 / 0.93 | 0.08 / −0.19 / 0.94 | 0.11 / −0.47 / 0.89 |
| FedKPer λ=10 + logit | −0.01 / −0.39 / 0.94 | −0.01 / −0.44 / 0.94 | −0.01 / −0.36 / 0.89 |
| FedBABU | 0.64 / 0.38 / 0.89 | 0.58 / 0.38 / 0.90 | 0.84 / 0.81 / 0.84 |
| FedRoD | 0.01 / −0.38 / 0.81 | 0.05 / −0.16 / 0.88 | 0.08 / −0.35 / 0.76 |

FedProx and Ditto sit on FedAvg. FedKPer λ=0 does too. λ=1 is between λ=0 and λ=10. Those four are on the figure for Blood and OrganC and were not rerun on CIFAR-10. CIFAR FedKPer's Whispers mean is pulled by seed 1 (R² 0.78); seeds 0 and 42 are 0.30 and 0.47.

## Protocol

Dirichlet α = 0.1, 10% participation, 5 local epochs, SGD, learning rate 0.01, momentum 0.9. BloodMNIST: 20 clients, 8 classes. OrganCMNIST: 30 clients, 11 classes. CIFAR-10 grid: 20 clients, 40 rounds, unless a sentence says 100 clients and 100 rounds. Model: the small CNN in this repo (`FedAvgNetCIFAR`: two convolutions and a linear layer; MedMNIST uses the same family). It is not a ResNet. The CNN has no BatchNorm.

FedKPer numbers are aggregation mode `infer`: the server estimates the mix from the upload and weights the average by that estimate. That estimate is itself a property-inference channel. FedLC trains cross-entropy on logits + log π and adds log π again only when the client scores its own model. FedPer averages the trunk and leaves the classifier on the client. The saved upload has the global head put back. FedBABU freezes the classifier, averages the body, and fits the classifier once after the last round. FedRoD, as trained, puts log π on the sum of the shared head and the personal head. The personal head stays on the client.

The channel run saves, for every participation, the classifier change against the model the client received (`head/round_*_client_*.npz`). Older runs did not. Their last upload minus the final global model is the wrong vector: for the last round that final model is the average of the two uploads, so the two clients partly cancel.

## Threat model

Honest-but-curious server. No secure aggregation in the runs. The global model sent down is the honest aggregate. A malicious server that sends weights chosen to amplify leakage is out of scope. The upload matters because some methods use the per-client update, and a later seismic operator may need to know which site an update came from.

## Utility

Macro-F1 is computed on classes present in the client's train mix (π > 1e-3), scored on the holdout, then averaged over clients. FedLC is scored with log π at inference. FedPer, FedBABU, and FedRoD are scored from the saved personal head. The FedRoD row is the recipe whose accuracy is fixed by adding log π at scoring time. It is not the paper's training split.

| Method | Blood local / global F1 | OrganC | CIFAR-10 |
|---|---|---|---|
| FedAvg | 0.73 / 0.53 | 0.83 / 0.76 | 0.60 / 0.29 |
| FedKPer λ=10 | 0.80 / 0.65 | 0.85 / 0.81 | 0.65 / 0.46 |
| FedPer | 0.81 / 0.73 | 0.83 / 0.81 | 0.65 / 0.55 |
| FedLC | 0.84 / 0.75 | 0.86 / 0.84 | 0.66 / 0.59 |
| FedKPer λ=10 + logit | 0.82 / 0.76 | 0.86 / 0.81 | — |
| FedBABU | 0.76 / 0.54 | 0.81 / 0.73 | 0.56 / 0.27 |
| FedRoD | 0.77 / 0.77 | 0.83 / 0.85 | 0.62 / 0.60 |

Three seeds and no intervals cannot settle a close comparison. CIFAR-10 FedAvg seed 1 is a chance global model (official test 0.12). Official test accuracy, 20 clients and 40 rounds: FedAvg 0.33, FedKPer 0.52, FedPer 0.62, FedLC 0.67. The 100-client, 100-round run, official test only: FedAvg 0.58, FedKPer 0.59, FedPer 0.60, FedLC 0.66. Do not compare either number to a paper unless client count, α, participation, local epochs, rounds, and architecture all match.

## What we are not claiming

FedLC does not hide which classes a client has. The row-size clue, read against the model the client received, ranks presence at AUC 0.99 on Blood.

The defense is one recipe, not a search. Eight images per class, inverse-frequency weights, and log π on the client. It was run on Blood only.

FedPer's body has not been shown to hide the mix. The probe we ran failed the FedAvg check, so it cannot be used to clear FedPer.

FedRoD on the figure is not the paper's training procedure.

No ε. The secure-aggregation number above is an information bound from the true mixes, not a protocol. Image recovery stays out of the class-mix claim until it is a real sample of images, attacked on the five-epoch update. Seismic is not started.

## Direction

The classification chapter is this mechanism. The learned attacker has been run, and it does not change the conclusion.

Logit adjustment closes the bias clue and leaves the row-size clue open. The combined defense reverses the bias clue instead of closing it. An attacker fit on shadow clients, who is free to use both clues and to flip a sign, still tells an absent class from a high-share class about 85% of the time on that defense, and about 87% of the time on FedLC.

Smoothing the prior does not produce a private operating point. On Blood the best presence AUC stays at or above 0.82 for every λ from FedLC to FedAvg, and the absent-versus-high-share accuracy stays near 0.90. A new defense is worth training only if that test falls to chance. Nothing tried so far does, and the public-pad defense will not be iterated.

FedPer is not in the channel table because its classifier never leaves the client, so there is no row to read. The body probe was already run, and it could not read FedAvg, so a blank FedPer result would not mean the body is safe.

Seismic is the application of the same attacker. Few sites, so the score is the distinguishing game across rounds, not R² on five sites. Shadow clients should come from the other survey. FedAvg against FedBN against GroupNorm matters because a BatchNorm UNet uploads the site's running statistics, and FedBN keeps them local. The FedLC presence result now holds on Blood, OrganC, and CIFAR-10, each with five seeds. FedPer is the classification piece still missing from the channel table.

## 2026-09-28

Workstream C, written before the slope fit.

Prediction for the λ-slope test. The outcome is the slope of one regression: the classifier bias of the upload minus the bias of the model the client received, averaged over that client's participations, on the class share π. Clients and classes are pooled. At λ = 0 the slope should be about 0. It should increase linearly with λ, and the sign should be positive, because the update moves with π − 1/K. λ = 1 is ordinary training. That slope has to come out clearly above 0 before any other λ is interpreted. The absolute size is not part of the prediction. Momentum and five local steps rescale a one-step gradient.

Outcome, same day. The positive control passed. At λ = 1 the slope is positive and the interval excludes 0 on all three datasets: Blood +0.344 [+0.287, +0.408], OrganC +0.184 [+0.161, +0.207], CIFAR-10 +0.483 [+0.436, +0.529]. The form is slope × (π − 1/K). The intercept after that centering is 0.000 at every point.

At λ = 0 the slope is two orders of magnitude smaller than at λ = 1, and it is slightly negative. The interval excludes 0: Blood −0.0027 [−0.0056, −0.0004], OrganC −0.0044 [−0.0087, −0.0004], CIFAR-10 −0.016 [−0.024, −0.007]. "About 0" is right next to the FedAvg slope. It is not an interval that contains 0.

On Blood the slope rises with λ. A straight line has slope +0.349 [+0.292, +0.410] and R² 0.986 [0.975, 0.995]. The fit is not uniform. Dividing each slope by λ gives about +0.21 at λ = 0.02, 0.1, and 0.3, and +0.34 at λ = 1. The λ = 0.3 point sits 0.030 below the line. The figure is `results/figures/lambda_slope.png`. The table is `results/tables/lambda_slope.csv`.

The three-dataset presence numbers are the same point estimates already in the table above. What is new is a client bootstrap, 1000 draws, clients resampled inside each seed, and the prior-only floor. That floor scores every client with the population mix. It is about 0.55 on Blood and OrganC and 0.51 on CIFAR-10, above 0.50, because a class that is common in the population is present more often.

| Dataset | FedLC row AUC | FedLC bias AUC | FedAvg row AUC | FedAvg bias AUC |
|---|---|---|---|---|
| Blood | 0.977 [0.965, 0.989] | 0.499 [0.472, 0.525] | 0.696 [0.661, 0.733] | 0.822 [0.795, 0.855] |
| OrganC | 0.997 [0.994, 1.000] | 0.471 [0.446, 0.496] | 0.732 [0.712, 0.754] | 0.822 [0.802, 0.847] |
| CIFAR-10 | 0.988 [0.983, 0.993] | 0.528 [0.500, 0.557] | 0.717 [0.687, 0.746] | 0.866 [0.846, 0.890] |

Leakage is the AUC minus the floor. FedLC's row leakage is +0.43 [+0.35, +0.44] on Blood, +0.45 [+0.40, +0.46] on OrganC, and +0.48 [+0.39, +0.48] on CIFAR-10. FedLC's bias leakage is −0.05 [−0.13, −0.03] on Blood and −0.08 [−0.14, −0.06] on OrganC, so those intervals do not rise above the floor. On CIFAR-10 the bias leakage interval includes 0. FedAvg's bias leakage is about +0.27 to +0.35 and its interval excludes 0 on every dataset. That is the positive control for this table.

Local accuracy was recomputed from the saved uploads and matched the stored means to four digits. Client intervals: Blood FedLC 0.932 [0.915, 0.946] and FedAvg 0.905 [0.886, 0.922]; OrganC FedLC 0.935 [0.916, 0.952] and FedAvg 0.917 [0.895, 0.935]; CIFAR-10 FedLC 0.886 [0.870, 0.901] and FedAvg 0.852 [0.831, 0.871]. The file is `results/tables/three_dataset.csv`.

The same bins, scored with the saved channels rather than the learned attacker. A class is compared with absent when its share is in (0.001, 0.03], (0.03, 0.075], (0.075, 0.175], or at least 0.25. FedAvg's bias is the control, and it reads the large shares: AUC 0.99 / 0.97 / 1.00 for a share of at least 0.25 on Blood, OrganC, and CIFAR-10. A share near 0.01 is close to the floor on Blood (bias AUC 0.61, floor 0.55). FedLC's row size stays high in every bin, including the rare one: Blood 0.978, 0.977, 0.991, 0.971. FedLC's bias does not. The learned attacker on these bins has not been refit yet. The hand-built file is `results/tables/share_sweep_hand.csv`.

Prediction for Decaf, written before that score. FedAvg should be high on presence and on proportions. FedLC presence AUC should be at least 0.95, and FedLC proportions should sit at or near the floor. FedPer is N/A: the saved upload puts the global head back. FedKPer's existing runs did not save the classifier delta against the received model, so this pass does not include them. That is a gap in the grid, not a result.

Outcome, same day. The hard rule fails the positive control. After five local epochs with momentum, every classifier row has at least one positive entry, on FedAvg and on FedLC, on all three datasets. The rule therefore calls every class present. Its accuracy is the fraction of classes that are present, about 0.50 on Blood, 0.46 on OrganC, and 0.52 on CIFAR-10. That is not a reading of the mix, and FedLC is not judged by it.

The continuous score is the sum of the positive entries in the row, averaged over the client's participations. FedAvg is the control and it passes: AUC 0.986 [0.979, 0.994] on Blood, 0.995 [0.990, 0.999] on OrganC, 0.991 [0.986, 0.997] on CIFAR-10. The floors are 0.55, 0.55, and 0.51. FedLC is 0.968 [0.952, 0.983], 0.987 [0.978, 0.994], and 0.984 [0.979, 0.990]. The prediction of at least 0.95 holds. Leakage above the floor is +0.42 to +0.47. An absent FedLC row is not exactly zero. On Blood its mean positive sum is 0.015, against 0.62 for a class the client holds.

Proportions use a line from that positive sum to the class share, fit on shadow clients that train with the victim's loss from a received model. This is not the paper's one-class gradient basis. The hard rule never removes a class, because it never fires, so the line sees every class. FedAvg is again the control: R² +0.44 [+0.41, +0.48] on Blood, +0.43 [+0.40, +0.45] on OrganC, +0.67 [+0.65, +0.70] on CIFAR-10. Pearson is +0.85 to +0.89. The floor for this R² is 0.

FedLC proportions stay near that floor on Blood, R² +0.044 [+0.036, +0.054], and on OrganC, +0.087 [+0.076, +0.100]. The prediction holds there. It does not hold on CIFAR-10, where FedLC R² is +0.31 [+0.29, +0.33], about half of FedAvg. Pearson stays positive on all three (+0.50, +0.55, +0.71), so the score moves with the share even where the line is too flat to name the fraction. The table is `results/tables/decaf.csv`.

The learned attacker was refit from the received models, 48 shadows, the victim's loss. FedAvg's mix is recovered first: Pearson +0.78 / +0.75 / +0.85 and leave-one-out R² +0.59 / +0.55 / +0.71 on Blood, OrganC, and CIFAR-10. The absent-versus-at-least-0.25 accuracy is Blood 0.89 / 0.92, OrganC 0.93 / 0.93, CIFAR-10 0.96 / 0.97 for FedLC / FedAvg. The previous received-model table was 0.88 / 0.92, 0.94 / 0.93, 0.96 / 0.97. Those point estimates are unchanged.

On the centred bins the same attacker is weak where a class rarely lands in the window, and some seeds have no class with enough examples. Averaging the seeds that do, a share near 0.01 has FedAvg AUC about 0.51, 0.51, and 0.51. That attack does not recover ordinary training on a 1% share, so FedLC is not judged by it. The hand-built row size is the score that does: FedLC AUC 0.978, 0.998, and 0.973 on that bin, against FedAvg 0.48, 0.57, and 0.49. The per-seed file is `results/tables/share_sweep.csv`.

The first client-bootstrap file for these bins was discarded. Each draw took a new sample of features and, separately, a new sample of labels, so the interval described a broken pairing and sat near 0.5 under every point estimate. The point estimates above are unaffected. The corrected file is `results/tables/share_sweep_intervals.csv`. Each point now sits inside its interval.

The per-class learned attacker still fails the FedAvg control on a share near 0.01. Its AUC is 0.512 [0.442, 0.587] on Blood, 0.506 [0.467, 0.559] on OrganC, and 0.509 [0.460, 0.587] on CIFAR-10. Those intervals cover chance, so FedLC is not judged by this attacker on that bin. The absent-versus-at-least-0.25 game does recover FedAvg, and the FedLC accuracies above keep their intervals: Blood 0.893 [0.855, 0.923], OrganC 0.934 [0.907, 0.949], CIFAR-10 0.958 [0.930, 0.976]. FedAvg on the same game is 0.920 [0.891, 0.943], 0.932 [0.909, 0.947], and 0.967 [0.942, 0.984].

A pooled attacker, which uses one logistic regression on the class's own row size and bias, is a different control. On a share near 0.01 it clears the floor on OrganC, AUC 0.635 [0.594, 0.678] against 0.551, and on CIFAR-10, 0.704 [0.657, 0.753] against 0.512. FedLC on those two is 0.984 [0.972, 0.993] and 0.969 [0.957, 0.981]. On Blood the same FedAvg interval includes the floor, 0.602 [0.538, 0.666] against 0.551, so the Blood FedLC pooled number is not used. From a share near 0.05 upward the pooled FedAvg control clears the floor on all three datasets, and the pooled FedLC AUC stays above 0.96.

Prediction for the body probe, written before that run. The saved `local/client_*.pt` is the upload after local training. Presence should sit above the prior-only floor for FedAvg, FedBABU, and FedPer. The size of the gap is not part of the prediction. FedAvg has to clear the floor before the other two are interpreted. Where a received model was saved, the score is per-class recall of the uploaded body minus recall of the body the client was sent. Otherwise the score is recall with the client mean removed.

Outcome, same day. The positive control failed. FedPer and FedBABU were not scored. The table `results/tables/body_probe.csv` has FedAvg only.

The score is the difference of per-class recall. Blood is 0.551 [0.510, 0.587] against a floor of 0.551. OrganC is 0.561 [0.531, 0.588] against 0.551, and that interval includes the floor. CIFAR-10 is 0.652 [0.617, 0.685] against 0.512, so presence does clear the floor there, with Pearson +0.20 [+0.15, +0.25]. Leave-one-out R² on CIFAR-10 is +0.034 [+0.020, +0.058]. On Blood and OrganC the R² interval includes 0. The control required every dataset, so the CIFAR-10 gap is not used to read the other two methods. This does not change the classifier result. It says this recall difference does not read an ordinary-training body on Blood or OrganC.

## 2026-09-29

Workstream A of the how-much plan, written before the rescore. The observer is the server looking at one client's update.

Prediction. FedAvg should be high on support and on counts. FedLC should stay high on support. Its counts should be only partly leaked: logit adjustment removes the bias term, and the row sizes still grow with the class count, so the counts R² should sit above the floor and below FedAvg. Dataset size should leak under both, because the whole update gets larger when the client has more data. The floor for counts is the per-class mean count, and the floor for size is the mean dataset size. Both of those R² numbers are already the amount above the floor.

FedKPer is not in this pass. Those runs did not save the classifier change against the received model. FedPer's body is not interpreted unless the saved FedAvg body probe clears the support floor, which it did not on Blood or OrganC.

Outcome, same day. The observer is server-individual. The table is `results/tables/howmuch.csv`. The figure file `results/figures/support_vs_howmuch.png` was first drawn with the joint ridge on the counts axis. On OrganC that FedLC interval runs down to −2.02, and that first panel stopped at −0.55. The file was replaced later the same day; the replacement is described below. The ridge numbers in this paragraph are from that first fit.

The count regressor recovers FedAvg. Counts R² is 0.261 [0.175, 0.275] on Blood, 0.297 [0.168, 0.378] on OrganC, and 0.353 [0.279, 0.365] on CIFAR-10. FedLC is not a partial leak on Blood or OrganC: 0.046 [−0.146, 0.099] and −0.333 [−2.019, 0.215], and both intervals include the floor. On CIFAR-10 it is partial, 0.280 [0.164, 0.312], below FedAvg and above the floor. The prediction of a partial leak on every dataset does not hold.

Row size is the support channel, and it does not hide counts either. Its FedAvg counts R² is above the floor on every dataset, including Blood at 0.167 [0.083, 0.185]. FedLC is at least as high: Blood 0.211 [0.115, 0.256], OrganC 0.268 [0.072, 0.355], CIFAR-10 0.399 [0.328, 0.409]. Support leakage for that channel stays high under FedLC, 0.426 [0.353, 0.446], 0.446 [0.394, 0.462], and 0.476 [0.392, 0.478]. At a 10% false-positive rate the true-positive rate is 0.98, 1.00, and 0.96, against floors of 0.06, 0.11, and 0.05. On Blood the λ curve moves support leakage from 0.43 at FedLC down to 0.15 at FedAvg, while this counts R² stays between 0.14 and 0.21.

Decaf's positive-sum line also recovers FedAvg counts: 0.307 [0.220, 0.343], 0.383 [0.288, 0.486], and 0.447 [0.377, 0.475]. FedLC is lower on Blood, 0.174 [0.071, 0.210], and on OrganC, 0.223 [0.128, 0.278], and essentially tied on CIFAR-10, 0.426 [0.349, 0.440] against 0.447. Told the true support, that line's proportions R² under FedLC is 0.101 [0.020, 0.186], 0.184 [0.111, 0.253], and 0.388 [0.335, 0.438]. The bias line, told the same support, sits on the floor under FedLC.

Dataset size does not leak. No attacker has a FedAvg size R² whose interval excludes 0 on all three datasets. The prediction that the whole update grows with n is not supported.

The pooled mix R², which mixes support and proportions, is higher than the counts R². For the row size under FedLC it is 0.37, 0.49, and 0.49. That number is not the how-much axis.

## 2026-09-29, expert correction

The joint ridge is not used as evidence that FedLC hides the counts. It scores below the row-size line under FedLC, and that line already recovers FedAvg. The counts axis of `results/figures/support_vs_howmuch.png` is now the best single channel whose FedAvg interval sits above the floor: the bias, the row size, or Decaf's positive sum. The channel is written on the point. Support leakage is still the row size. The choice is in `results/tables/support_vs_howmuch_attackers.csv`.

On Blood the best channel is the row size at FedLC, 0.211 [0.115, 0.256], and Decaf from λ = 0.02 through FedAvg: 0.219 [0.120, 0.253], 0.231 [0.138, 0.266], 0.224 [0.130, 0.255], and 0.307 [0.220, 0.343]. On OrganC it is the row size for FedLC, 0.268 [0.072, 0.355], and Decaf for FedAvg, 0.383 [0.288, 0.486]. On CIFAR-10 it is Decaf for both, FedLC 0.426 [0.349, 0.440] and FedAvg 0.447 [0.377, 0.475].

Client dataset size is not nearly constant, so the null on size is not an empty comparison. Within a seed the coefficient of variation of n is about 1.03 on Blood, 0.87 on OrganC, and 0.76 on CIFAR-10. A Blood client can have a handful of training images or about two thousand. CIFAR-10 runs from about ten to several thousand. No extra run with forced size variation is needed for that null.

These runs do not hand the server n_i. Aggregation is equal weights. Standard FedAvg does send n_i as the aggregation weight. Unless that weight is computed inside the secure sum, the protocol leaks the dataset size even when the update itself does not.

Prediction for the pooled two-channel ridge, written before that fit. The published ridge maps every class's row size and bias onto the count vector. It does not contain the row-size line as a special case, which is why it can land below that line. The refit uses only that class's own row size and bias, pooled across classes the same way as the line, with the two features standardized and the penalty chosen on the shadows. If the bias carries nothing, this fit should sit next to the row-size line. A gap below that line is a diagnostic of the fit, not a hiding of the counts.

Outcome, same day. The refit sits on the row-size line when the bias has nothing to say, and above it when the bias carries the counts. Under FedAvg the gap, pair minus row, is +0.119 [+0.099, +0.154] on Blood, +0.081 [+0.051, +0.128] on OrganC, and +0.174 [+0.156, +0.215] on CIFAR-10. Under FedLC the gap is −0.013 [−0.027, +0.001] on Blood, −0.009 [−0.071, +0.031] on OrganC, and +0.029 [−0.016, +0.061] on CIFAR-10. Those three intervals include 0. The one place the interval sits below 0 is Blood at λ = 0.02, gap −0.015 [−0.030, −0.006]. The penalty chosen on the shadows is large there (median α about 32), so the row coefficient is shrunk relative to the unpenalized line. That is the diagnostic. It is not the joint ridge's collapse: that fit's FedLC counts R² on Blood was 0.046, and this one is 0.198 against a row-size line of 0.211. The table is `results/tables/pair_ridge.csv`.

Prediction for the FedKPer rerun, written before training. Blood only, five seeds, the same 40 rounds and Dirichlet split as the scored runs, λ cap 10, diversity inferred from the public probe. The classifier change is saved against the model the client received. Two losses: plain cross-entropy, and logit adjustment. The row-size count attacker has to keep recovering the existing FedAvg runs before either FedKPer loss is interpreted. Without logit adjustment, row-size counts and support should look like FedAvg. With logit adjustment, support should stay high, the bias channel should drop, and the row-size count R² should stay above the floor. The distillation term may shrink the update. The prediction does not say by how much.

Outcome, same day. The row-size attacker already recovered FedAvg, so these Blood numbers are interpreted. Shadows train with FedKPer's own loss, including the distillation, from a received model. The table is `results/tables/fedkper_howmuch.csv`.

Without logit adjustment, FedKPer looks like FedAvg. Row-size counts R² is 0.168 [0.071, 0.189], against 0.167 for FedAvg. Bias counts R² is 0.270 [0.179, 0.310]. Row-size support leakage is +0.178 [+0.099, +0.208].

With logit adjustment the bias channel shuts, as it does for FedLC. Bias counts R² is −0.033, and the interval [−0.133, −0.022] does not reach the floor (unshrunk fit; recalibrated in the Oct-14 entry). The point is 0.011 above the top of that interval. Bias presence AUC is 0.473 [0.437, 0.510], against a floor of 0.551. The row size does not shut. Its counts R² is 0.282 [0.185, 0.330], above FedAvg and above FedLC on this dataset. Support leakage on that channel is +0.307 [+0.227, +0.338], AUC 0.858 [0.831, 0.890]. That is above the floor and below FedLC's row AUC of 0.977. Told the true support, the row-size line still reads the fractions, R² 0.362 [0.304, 0.416], and the bias line does not, 0.040 [−0.074, 0.138]. Decaf's count interval includes the floor, 0.068 [−0.031, 0.090], so that channel is not used for the counts. Its presence AUC stays above the floor, 0.816 [0.784, 0.850].

Dataset size is not interpreted. The same attackers do not read n under FedAvg on every dataset. Local accuracy, the mean of the five seed means, is 0.924 without logit adjustment and 0.940 with it. FedLC on Blood was 0.932.

The distillation term does not hide the counts. It leaves the row-size channel at least as open as ordinary training, and it softens the presence signal relative to FedLC without bringing it down to the floor.

Prediction for the FedPer body attacker, written before that run. The score is the change, per class, in the mean public feature between the body the client received and the body it uploaded. A line from that shift to the class count is fit on shadows, not on the client being scored. FedAvg has to clear the counts floor on every dataset before FedPer is read. If it does not, FedPer stays unmeasured.

Prediction for balanced gradients, written before that run. One step is the mean of the per-class mean gradients, over the classes the client holds, and the class prior is added only at inference. At one step the row-size counts R², the size R², and the fractions given the support should sit near their floors, and support leakage should stay high. At two steps and at five steps the row-size counts should move back toward ordinary training. The comparison is count-weighted gradients at one, two, and five steps, plus the existing FedAvg and FedLC runs. The row-size count attacker is the one that has to recover FedAvg before the balanced numbers are interpreted. Blood first.

## 2026-10-03

Outcome for the FedPer body attacker. The score is the per-class move of the mean public feature from the body the client received to the body it uploaded. The line to the class count is fit on shadows. The observer is server-individual. The table is `results/tables/feature_shift.csv`.

Blood FedAvg counts R² -0.038 [-0.135, -0.049].
CIFAR-10 FedAvg counts R² -0.026 [-0.111, -0.048].
OrganC FedAvg counts R² -0.016 [-0.086, -0.010].

The FedAvg control did not pass on every dataset. FedPer stays unmeasured. This does not change the classifier result.

## 2026-10-07

Outcome for balanced gradients on Blood, workstream B. BloodMNIST, Dirichlet α = 0.1, 20 clients, 40 rounds, 10% participation, SGD lr 0.01 momentum 0.9. Five seeds (0, 1, 2, 3, 42). Each client sends a fixed number of full-dataset steps: `example` is the usual count-weighted mean gradient, `class` is the mean of the per-class mean gradients over the classes the client holds. Log π is added only at local evaluation for the class arm. Observer is server-individual. Shadows (48 per run) train with the victim's trainer from a received model saved during the run. Table is `results/tables/balanced_howmuch.csv`. Per-run predictions are under `results/attacks/howmuch/balanced_blood/`.

The count-weighted control passes, so the balanced arm is interpreted. Row-size counts R² for `example`: 0.215 [0.123, 0.263] at 1 step, 0.178 [0.081, 0.218] at 2 steps, 0.132 [0.034, 0.160] at 5 steps. Bias counts R² stays high for the control: 0.286 / 0.282 / 0.252. Decaf counts R²: 0.268 / 0.248 / 0.226.

Under class balancing the counts sit on the floor at every step count. Row-size counts R²: −0.054 [−0.151, −0.066] at 1 step, −0.051 at 2 steps, −0.052 at 5 steps. Decaf counts R²: −0.013 / −0.016 / −0.022. Bias counts R²: 0.021 / 0.019 / 0.012, intervals covering the floor. Dataset size does not leak under either arm: every size R² point is negative. The prediction that balanced counts sit at the floor holds; the prediction that 2 and 5 steps move back toward ordinary training does not hold on Blood.

Support behaves differently per channel. The bias and Decaf presence scores stay at 0.99 AUC (TPR at 10% FPR 1.00) under class balancing at all three step counts, against a floor of 0.55. The row-size presence AUC under balancing is 0.51 / 0.46 / 0.53, intervals covering the floor. So the bias channel still names which classes are present even when every present class gets equal weight: the sign pattern of the update says which classes pulled, while the row magnitudes no longer say how hard. Told the true support, the fractions are not distinguishable from the floor under balancing: oracle proportions R² is 0.003 / 0.003 / 0.032. The estimated-support proportions are not distinguishable from the floor either (−0.03 to −1.21 with the unshrunk fit; recalibrated in the Oct-14 entry), because the support estimate from these channels is wrong often enough to hurt.

Local accuracy (mean of client means, `infer_log_prior` on for the class arm): balanced 0.688 / 0.703 / 0.790 at 1 / 2 / 5 steps; count-weighted 0.461 / 0.587 / 0.730. The global model on client holdouts is 0.247–0.357. Accuracy-vs-rounds from `history.jsonl`: balanced-5 reaches local 0.60 by round 40 with global 0.35; balanced-1 ends at local 0.16, global 0.21. One balanced step barely trains; five steps train but still sit below the FedLC Blood local mean of 0.93. Balanced gradients hide "how much" on Blood at the cost of the model: at 1 step counts are at floor and the model is near chance, at 5 steps counts are still at floor but local accuracy is 0.79 vs 0.93 for FedLC.

Deviation from the Oct-3 prediction: support was predicted to stay high without naming a channel. It stays high on the bias/Decaf channels and falls to floor on the row-size channel. Counts were predicted to return toward ordinary training at 2 and 5 steps. They do not on Blood. Next: OrganC and CIFAR-10 grids, then the support-vs-howmuch point for balanced-5 (row-size counts channel, which passed its FedAvg control) once all three datasets are scored.

## 2026-10-07, OrganC balanced outcome

Same protocol on OrganCMNIST (30 clients, 11 classes), shadows from received models, table `results/tables/balanced_organc_howmuch.csv`. The count-weighted control passes on every channel: row-size counts R² 0.365 [0.290, 0.417] / 0.285 / 0.264 at 1 / 2 / 5 steps, bias 0.476 / 0.468 / 0.392, Decaf 0.429 / 0.370 / 0.377.

Under class balancing the counts are not distinguishable from the floor at every step count, same as Blood. Row-size counts R²: −0.044 / −0.041 / −0.033. Decaf counts R²: 0.007 / −0.002 / 0.014. Bias counts R²: 0.055 / 0.052 / 0.038, the 1-step interval [−0.013, +0.072] covering the floor. Size R² is at the floor on every arm. Oracle proportions under balancing: −0.001 / −0.005 / −0.017. Estimated-support proportions are not distinguishable from the floor.

Support under balancing: bias AUC 1.00 / 1.00 / 0.99 (TPR at 10% FPR 1.00 / 1.00 / 0.98), Decaf AUC 1.00 at all three steps, against a floor of 0.55. Row-size support AUC is 0.56 / 0.57 / 0.75: the 5-step point [0.72, 0.78] sits above the floor, so unlike Blood the row magnitudes at 5 steps carry some presence signal on OrganC. Local accuracy: balanced 0.667 / 0.721 / 0.834; count-weighted 0.371 / 0.632 / 0.780. Balanced-5 trains (0.83) but stays below the FedLC OrganC local mean of 0.94.

The Blood conclusions replicate: class balancing closes the counts at every step count while bias/Decaf support stays at ceiling. The one dataset difference is row-size support at 5 steps, which lifts off the floor on OrganC and not on Blood. Next: CIFAR-10 (training).

## 2026-10-08, CIFAR-10 balanced outcome

Same protocol on CIFAR-10 (20 clients, 10 classes), shadows from received models, table `results/tables/balanced_cifar_howmuch.csv`. The count-weighted control passes on every channel: row-size counts R² 0.330 [0.233, 0.374] / 0.282 / 0.237 at 1 / 2 / 5 steps, bias 0.429 / 0.423 / 0.410, Decaf 0.404 / 0.382 / 0.361.

Under class balancing the counts sit on the floor at every step count, all three datasets now agree. Row-size counts R²: −0.043 / −0.043 / −0.044. Decaf: −0.010 / −0.009 / −0.002. Bias: 0.034 / 0.033 / 0.028, intervals covering the floor. Size R² is negative on every arm (about −0.44). Oracle proportions under balancing: 0.006 / 0.009 / 0.006. Estimated-support proportions negative.

Support under balancing: bias AUC 0.95 at all three steps (TPR at 10% FPR 0.85 / 0.86 / 0.89), Decaf AUC 0.94–0.95, against a floor of 0.51. Row-size support AUC is 0.52 / 0.54 / 0.47, intervals covering the floor. Local accuracy: balanced 0.657 / 0.672 / 0.712; count-weighted 0.325 / 0.485 / 0.611. Balanced-5 trains (0.71) but stays below the FedLC CIFAR-10 local mean of 0.89.

Workstream B closes on classification: one balanced step per round hides counts, size, and fractions on all three datasets while bias/Decaf support stays near ceiling. The multi-step prediction fails everywhere — 2 and 5 balanced steps do not move counts back toward ordinary training. The cost is utility: balanced-5 local accuracy trails FedLC by 0.14 (Blood), 0.10 (OrganC), 0.18 (CIFAR-10). The support-vs-howmuch point for balanced-5 uses the row-size counts channel (its FedAvg control passes on all three datasets) against row-size support; the bias-support number (0.95–1.00) is reported alongside, not on the axis. Next: workstream C sums.

## 2026-10-15, balanced-minibatch outcome (expert issue B)

Table `results/tables/balanced_mb_howmuch.csv` (Blood, 5 seeds, 5 epochs class-balanced minibatches, log π at inference only). Go/no-go on counts: row +0.011 [−0.085,−0.000], decaf +0.076 [−0.009,+0.076], bias +0.039 — all at the floor, inside the full-batch intervals. Counts do NOT come back under realistic training. GO on the headline.

Two qualifications, both new vs full-batch. (1) Dataset size leaks through the bias magnitude: size R² +0.446 [+0.308,+0.519] (full-batch: negative everywhere). Minibatch epochs do compute ∝ n (n/50 batches × 5), while full-batch steps are fixed compute — the update grows with the dataset. (2) Told the true support, fractions partially leak: oracle proportions R² +0.162 (decaf) / +0.144 (bias), intervals above 0 (full-batch: ≈0). Support stays high but softer: bias AUC 0.83 (was 0.99), decaf 0.90 (was 0.99), row 0.71 (was 0.51).

Utility vindicates the budget-suspicion: local accuracy 0.930, level with FedLC (0.932) and far above full-batch balanced-5 (0.790). The "cost" was an optimization-budget artifact. Net: balanced minibatches hide exact counts at FedLC-level utility, but leak dataset size and partial fractions — "how much" is narrowed, not closed. The support-vs-howmuch point for balanced-mb uses the row counts channel; the bias-size leak is reported alongside.

## 2026-10-15, λ-at-sum outcome (N=3, Blood)

Table `results/tables/sums_lambda_blood.csv`, trend `sums_lambda_trend.csv` (Spearman over 3 λ points — direction only, coarse). Individual bias counts R²: −0.04 (λ=0) → +0.15 (λ=0.3) → +0.31 (λ=1); the bias channel opens with λ under the sum observer's victims too. Individual row counts flat: 0.49/0.47/0.45. Server-sum counts on the floor at all three λ (−0.03 to −0.02). Server-sum row presence declines with λ: 0.67 → 0.57 → 0.51 — FedLC's union signal fades as smoothing reopens the bias channel, and at λ=1 the sum reads nothing. Peer row counts flat at +0.39–0.42: at N=3 a peer reads the others' mean counts regardless of λ. The sums track personalization strength exactly as the individual observer does, compressed to the floor on counts.

## 2026-10-07, workstream C prediction

Written before any sum code runs. Observer is server-sum (sees only the mean update of the round's members) or peer-sum (sees the sum and subtracts its own update). Setup: full participation, N in {2, 3, 5, 8}, α in {0.1, 0.5}, Blood first. The sum is simulated in plaintext from saved per-client updates; no crypto runs.

Prediction. Counts R² under server-sum should dilute roughly as 1/m for m members: a member's count signal is averaged with m−1 others, so the per-client line fit on the sum should sit near floor already at N=3 and stay there. Peer-sum recovers one equation in m unknowns per round, so after enough rounds a peer accumulates a linear system on the cohort's counts; with full participation and 40 rounds the peer-sum counts R² should sit between server-individual and server-sum, closer to individual for N=2 and near sum for N=8. Support under server-sum becomes the union across members: presence AUC for a single member's support, scored from the sum, should fall toward the floor faster than counts do, because one present member lights the class for the whole sum. FedLC vs FedAvg ordering should survive the sum: whatever leaks individually leaks more in the sum, so the method ranking on counts and support is preserved with compressed gaps. The fingerprint sub-task (10 clients, 3 per round, hidden schedule): peers should recover their cohort from zero patterns under FedAvg, and worse under FedLC only where FedLC's row-size channel already fails individually — on Blood the bias channel still reads support, so the fingerprint should work under both.

## 2026-10-09, workstream C status (no outcome yet)

Grid training in progress, no scores to report. `configs/sweeps/sums_blood.json` (FedAvg/FedPer, 16 jobs x 20 seeds = 320 runs) and `configs/sweeps/sums_blood_fedlc.json` (FedLC, 8 jobs x 20 seeds = 160 runs): N in {2, 3, 5, 8}, α in {0.1, 0.5}, full participation, 40 rounds, `privacy_save_received` on. At last check 59/320 and 58/160 `results.json` files. Pace is slow (~1.5 runs/hour per sweep) with another job sharing the RTX 3090; that job is expected to end soon, after which the sweeps get the full GPU.

New module `fedseismic/privacy/sum_channel.py` (plaintext sum simulation, no crypto): server-sum is the mean of members' mean head-delta channels, peer-sum subtracts the member's own update and scores the residual against the other members' aggregate counts, pooled across members within the seed. Count lines reuse the howmuch shadow fits (victim's trainer, received-model starts). Smoke test on one finished N=2 run: peer-sum equals server-individual exactly, as the algebra requires (residual is the other member's update). Production scoring uses 48 shadows per run and waits for the grid. Also fixed in passing: sweep `_cell_config` let spec-level `num_clients`/`sample_ratio` override job-level silo counts; job level now wins, and the silo count is part of the job name.

## 2026-10-13, workstream C shadow diagnosis (no outcome yet)

First full scoring (`results/tables/sums_blood.csv`, now superseded) failed its own control: individual counts R² was negative at every N (−1.1 at N=2, −0.06 at N=8), against the 20-client FedAvg control of +0.17. Two compounding causes, both confirmed with numbers:

1. Shadow scale mismatch. Shadows split the pool 48 ways (mean size ~250) while N=2 victims hold ~pool/2 (8594/973). The count line was calibrated 24x off scale. Fixed with `cohort_size`: shadows drawn from repeated Dirichlet(N) partitions (`meta_attacker._shadow_loaders`, `howmuch.shadow_counts`), same RNG stream so counts align with features (verified mismatch 0.0).
2. Regime mixing across rounds. Victim means average 40 rounds but are dominated by rounds 1–5 (mean |bias| L1 0.94 → 0.02 by round 5, ~0.008 by round 40; global acc 0.28 → 0.90). Shadows sampled uniform over 40 received models are mostly converged noise: single-seed test showed early-start shadows carry slope (pooled R² 0.41) and late-start shadows do not (0.12).

An external review (@advisor) corrected the first attempted fix (hard cutoff at round 10 on the shadow side only): the victim is a 40-round mean while a shadow is a single delta, so restricting only the shadows swaps one mismatch for another, and the cutoff was tuned after seeing the failure. Adopted fix: both sides live in the early regime. Victim channels average rounds 1–10 (`sum_channel.ROUND_LO/HI`, `participation_channels` window); shadows start in rounds 1–10 (`collect_shadows(start_round_max)`, cache-keyed). The window is justified by the label-free decay curve (server-visible magnitudes: ~0.9 round 1, ~0.03 round 10, ~0.008 round 40), fixed globally before per-cell scoring, not tuned per cell. The `cohort_size` fix is kept. Full-trajectory shadows (mean over all 40 starts per shadow) would be exactly like-for-like but cost 40x; the windowed statistic is the documented, cheaper attacker, and C-vs-B uses a stronger attacker than B — reported as such.

Validation on N=2 FedAvg α=0.1, 5 seeds, regime-matched both sides: individual row counts R² 0.172/0.137/0.148/−0.243/0.234, decaf 0.322/0.343/0.444/−0.068/0.595, bias −0.086/0.225/0.250/−0.312/0.257. Seed 3 fails on all channels (outlier; the 20-seed bootstrap absorbs it). Server-sum counts R² sits at −0.15 to −0.49: the dilution is visible even before CIs. Full 480-run rescore with the fixed attacker is running; the table will be overwritten only if the per-cell FedAvg individual control passes. FedPer stays not-measured on the head (upload carries the global head back, deltas zero by construction).

## 2026-10-14, workstream C outcome, Blood α=0.1

Table `results/tables/sums_blood_a01.csv` (20 seeds per cell, regime-matched shadows, rounds 1–10 victim means). Figure `results/figures/leakage_vs_silos.png`. FedAvg individual (server-individual) control passes on every channel at every N: row counts R² 0.13 / 0.38 / 0.46 / 0.54 at N=2/3/5/8, decaf 0.39 / 0.53 / 0.57 / 0.61, bias 0.14 / 0.28 / 0.36 / 0.39, all intervals above the floor. FedLC individual: row 0.23 / 0.41 / 0.48 / 0.46, decaf 0.14 / 0.25 / 0.31 / 0.32, bias at floor (−0.24 to −0.04), replicating the no-sum result that logit adjustment kills the bias channel.

Server-sum counts sit on the floor at every N under both methods: FedAvg row −0.19 / −0.10 / −0.08 / −0.05, FedLC row −0.21 / −0.10 / −0.07 / −0.05. Even N=2 kills "how much": the two members' count signals average out beyond what the single-client line can read. The 1/m dilution prediction holds in the strong sense — no gradual decay is visible because the floor binds already at N=2.

Server-sum support is the split result. FedAvg sum presence AUC sits at the floor (bias 0.51–0.57, row 0.51–0.57). FedLC sum presence via row sizes stays clearly above it: 0.67 / 0.67 / 0.65 / 0.62 at N=2/3/5/8, intervals excluding the floor. The union-support effect: one present member lights the class for the whole sum, and FedLC's row magnitudes preserve that light. The encrypted sum hides how-much but not support — and the method ranking reverses relative to the individual observer (FedLC leaks support through the sum while FedAvg does not).

Peer-sum counts: N=2 equals individual by algebra (verified: identical to 3 decimals). N=3 partially recovers the others' mean counts (FedAvg decaf +0.21 [0.16, 0.26], row +0.09 [0.04, 0.14]; FedLC row +0.11). N=5 and N=8 are negative: a peer cannot unmix the mean of 4–7 others with a single-client line. The prediction (peer between individual and sum, closer to individual at N=2) holds at N=2–3 and fails at N≥5 — the linear-system intuition ignored that the residual averages away the per-client scale the line was fit on.

Peer-sum support is the strongest leak in the grid. Others'-union presence AUC: FedAvg 0.92–1.00 on all channels at N=2/3/5; FedLC row/decaf 0.99–1.00 at N=2/3/5 with bias at chance (0.37–0.41, as logit adjustment predicts). At N=8 the union covers all classes (no negatives) so the metric is degenerate — union saturation, not a hiding result. A peer with the sum learns exactly which classes its cohort holds.

Prediction vs outcome: dilution-holds-strongly (stronger than predicted); union-support reverses the method ranking under the sum (predicted preserved ranking — wrong); peer-counts partial at N=3, dead at N≥5 (predicted gradual — wrong direction at large N); fingerprint sub-task still to run (its grid is the partial-participation analogue of this peer result). Next: α=0.5 rescore, fingerprint training (10 clients, 3/round), then the figure with both alphas.

## 2026-10-14, fingerprint outcome

Grid `results/runs/fingerprint_blood{,_fedlc}` (10 clients, 3/round, Blood α=0.1, 20 seeds each, FedAvg/FedLC), table `results/tables/fingerprint_blood.csv`. Pairwise cohort-membership AUC from residual cosine distance, no fitting: FedAvg bias 0.578 [0.436, 0.719], row 0.848 [0.789, 0.906], decaf 0.811 [0.667, 0.956]; FedLC bias 0.684 [0.386, 0.983], row 0.987 [0.974, 1.000], decaf 0.961 [0.939, 0.983]. Floor 0.5.

Peers recover their cohort under both methods on the row/decaf channels, and FedLC row (0.99) beats FedAvg row (0.85) — the same presence-signal ranking as the individual observer. The bias channel is weak (FedAvg interval touches chance) to uninformative (FedLC interval spans 0.39–0.98). The Oct-7 prediction (fingerprint works under both) holds for row/decaf; the bias-specific expectation does not resolve. Note the grid needed a partition-retry rule (`experiment._train_splits_or_retry`): extreme Dirichlet draws left a client empty (seed 2 crashed the train DataLoader); attempt 0 reproduces the legacy stream exactly so published splits are unchanged, later attempts use derived seeds with a warning, and `howmuch.real_counts` shares the helper so its label check stays in sync.

## 2026-10-14, workstream C outcome, Blood α=0.5

Table `results/tables/sums_blood_a05.csv` (20 seeds per cell, same regime-matched attacker). The individual row control fails at N=2 and N=3 (FedAvg −0.20 [−0.25,−0.16] / −0.06 [−0.09,−0.03]; FedLC −0.17 / −0.03): under mild skew two or three members hold similar mixes, so there is no count variance for the line to read within a seed. Those cells are not interpreted, per the control rule. The control passes at N=5 and N=8 (FedAvg +0.09 / +0.19, FedLC +0.12 / +0.22).

Interpretable cells (N=5,8) replicate α=0.1 exactly. Server-sum counts on the floor (FedAvg −0.15/−0.13, FedLC −0.15/−0.13). Server-sum row presence above the floor for FedLC (0.66/0.64) and at/below it for FedAvg (0.36/0.41 — the N=5 point sits below chance, the union of five mild-skew members carries no row-magnitude signal). Peer presence is NaN at N=5,8 (union saturation arrives sooner under mild skew); at N=2,3 it reads 0.95–1.00 but those cells are gated by the failed counts control, so they are reported only as raw numbers, not findings. The α=0.1 conclusions stand with no alpha interaction: sums hide how-much at every N, FedLC support survives the sum, peers read others' support. The main figure stays α=0.1 (all cells pass); α=0.5 is the robustness check.

## 2026-10-15, summary table (threat model × method × statistic)

BloodMNIST unless noted; 95% CIs in the cited tables. "Yes" = interval excludes the floor; "≈floor" = interval covers it; "partial" = above floor but well below the individual observer.

| Observer | Method | Counts R² | Size | Support (AUC) | Fractions | Utility (local) | Source |
|---|---|---|---|---|---|---|
| Individual | FedAvg | yes 0.17–0.33 | no¹ | yes 0.82–0.93 | yes | — | howmuch.csv |
| Individual | FedLC | yes 0.21–0.40 | no¹ | yes 0.98–1.00 | partial (oracle 0.10–0.39) | 0.93 Blood | howmuch.csv |
| Individual | Balanced-1-step | ≈floor | no¹ | yes 0.99 (bias/Decaf) / ≈floor (row) | ≈floor | 0.69–0.79 | balanced_*.csv |
| Individual | Balanced-minibatch | ≈floor Blood / partial 0.17 OrganC / leaks 0.20–0.36 CIFAR | **yes 0.45–0.78 (bias)** | yes 0.77–0.99 | partial–yes (oracle/est. 0.14–0.36) | 0.93 / 0.94 / 0.88 | balanced_mb_*.csv |
| Server-sum N=2–8 | FedAvg | ≈floor | — | ≈floor 0.51–0.57 | — | — | sums_blood_a01.csv |
| Server-sum N=2–8 | FedLC | ≈floor | — | **yes 0.62–0.67 (row)** | — | — | sums_blood_a01.csv |
| Peer-sum N=2–3 | FedAvg | yes (N=2 =indiv; N=3 partial) | — | yes 0.92–1.00 | partial (oracle, N≤3) | — | sums_blood_a01.csv |
| Peer-sum N≥5 | either | ≈floor | — | yes (union) | no | — | sums_blood_a01.csv |
| Fingerprint (10c, 3/r) | FedAvg / FedLC | — | — | cohort 0.85 / 0.99 (row) | — | — | fingerprint_blood.csv |
| Seismic (N=3–5, last round) | FedAvg / FedBN | bias-w Pearson +0.4–0.6; calibrated ≤+0.13 | — | bias bal-acc 0.76/0.90; BN top-1 0.98/0.94 | — | 0.82–0.91 (mIoU, present-class) | seismic_f3.csv |

¹ FedAvg's "Size: no" is update-magnitude only — standard FedAvg sends n_i as the aggregation weight, so size leaks through the protocol unless weights go inside the secure sum.

Wording audit: "floor" = per-class-mean R² 0, population-mix presence AUC, or 0.5 pairwise AUC as the metric defines; "closed/hides" = interval covers the floor; "survives/leaks" = interval excludes it; "partial" = excludes the floor but below half the individual-observer level. No "below the floor" anywhere.

## 2026-10-16, strict-FedBN and GroupNorm identity (free rescores)

Strict FedBN (BN keys stripped from the upload) needs no retrain: our FedBN aggregate already skips BN keys, so the trajectory is identical with or without them in the upload. Masking BN at scoring time gives the strict result with zero GPU: the BN stats attack is N/A by construction, and every head/outc number is unchanged (disjoint tensors). The practical Chapter-6 statement: stripping BN keys from uploads removes the statistics channel; everything else in the tables stands.

GroupNorm nearest-neighbor re-identification (no BN anywhere), 3 seeds each: GN affine weights top-1 0.97 FedAvg / 1.00 FedBN vs 0.33 chance; full-update sketch 0.93–0.98. Sites stay identifiable without statistics — learned normalization weights absorb site properties too. So BN statistics are sufficient but not necessary for site linkability; the consortium warning covers normalization weights generally. (The outc-bias NN check is meaningless — view files carry the reconstruct head; outc exists for round 20 only.)

## 2026-10-17, multisurvey go/no-go rule (pre-registered before any joint run finishes)

Solo_a (F3-alone) is done; solo_b and multisurvey have no completed runs. Per plan v3: GO iff (1) a personalization or collaboration gain on EACH survey's own holdouts, scored with present-class mIoU, and (2) multi-survey beats the solo baselines on at least one survey. Official-test F3 mIoU is reported alongside, not as the decision rule. (An earlier version of this entry made the rule official-test-based with a 0.03 margin and cut personalization-gap framing — those are thesis-framing decisions for Carlos, flagged here, not adopted.) If either condition fails, Parihaka goes to future work and Chapter 6 stays F3-only.

## 2026-10-19, Setup B interpretation (head vs body) + frozen-body ladder

Carlos: "if we trained a different classifier head in every model, it learns different WEIGHTS, doesn't it?" Yes — per-survey heads change body+head weights, so Setup B is not a clean "head conflict" isolation. Advisor ruling, with the restricted-argmax result (below) folded in:

- The restricted-argmax check ruled out **inference-time** 12-class softmax competition (F3 0.676 → 0.676; Parihaka 0.769 → 0.727). So a Setup B win is NOT "head conflict at inference."
- The real mechanism is **training-time** loss coupling: a shared 12-way softmax on disjoint labels pushes each client's cross-entropy to suppress the other survey's logits, sending gradient into the shared body. Per-survey heads (or masking the loss to own classes) remove that coupling.
- **Defensible claim:** "survey-specific output layers lower interference by removing the cross-survey negative-logit gradient." Not "head conflict at inference."
- Per-survey heads treat a **symptom** (loss coupling); the underlying finding is negative transfer from sharing parameters across disjoint-label tasks. Frame as a **decomposition**, not a fix.

**Frozen-body ladder (cheapest first, before Oct 25 freeze):**
1. **Frozen-body probe** (minutes): freeze the FedBN body (shared weights + own BN), train only fresh per-survey 6-class heads. ≈solo (0.80) → features fine, the problem is loss/head training. ≈multi (0.68/0.77) → body features degraded.
2. Full Setup B retrain — only if (1) leaves a large gap.
3. Solo bound.

**Not foreordained:** Setup B ≈multi → body-sharing is the problem (per-survey heads don't help); ≈solo → loss coupling is the mechanism; >solo → positive transfer. Report the ladder gaps as attribution.

**Advisor flag:** the F3 gap is mostly client 1 (0.44 vs client 0's 0.92) — intra-survey client heterogeneity, not cross-survey interference. Compare client 1's F3-solo score vs its multi score before blaming Parihaka for F3's 0.68.

## 2026-10-19, shared-body + per-survey heads (exploratory) — threshold pre-registered

Carlos: skip further 12-class runs; try shared body + different (per-survey) heads. Advisor ruling:

- **Drop the remaining 12-class FedBN fills. Keep the solo-B fills** (the baseline for any shared-body comparison — not really a 12-class run). The NO-GO is defensible without more FedBN seeds: vanilla FedAvg's Parihaka collapse (0.0) is a one-seed-shown mechanism finding, and FedBN 0.77 < solo 0.80 stands as "does not beat solo" even at 1 seed (report as single-seed, note 0.03 may be seed noise).
- **The shared-body run is EXPLORATORY only — it cannot rescue the multi-survey chapter.** It was designed after seeing the NO-GO, so it cannot reverse a pre-registered decision. Report as "evidence for the head-conflict mechanism / candidate future-work design," not a GO.
- **Pre-registered threshold (fixed before running):** the shared-body design "wins" as a mechanism signal only if it beats solo on BOTH surveys by more than the solo seed spread (solo A spread ≈ 0.79–0.81, so margin > 0.03). Anything less is "matches solo," and matches-solo does not reverse the decision.

**Zero-GPU check first (advisor):** re-evaluate the existing 12-class FedBN checkpoints with argmax restricted to each survey's own 6 classes (F3 = 0–5, Parihaka = 6–11). If the gap to solo closes, the 12-class softmax head-conflict is already shown at no compute cost. Code: `fedseismic/privacy/restricted_head_check.py`.

**Minimal clean shared-body setup:** copy the FedBN run exactly, change ONLY the final 1×1 classifier to be local + per-survey (6 classes); BN stays local (FedBN). Same seeds as solo for paired comparison. Report per-survey mIoU + per-class IoU for the gap-driving classes.

**Cut order if time runs short (before Oct 25 freeze):** 12-class FedBN fills first (cut by default) → optional training-time class-restriction ablation → shared-body seeds 1,2. Never cut solo-B fills.

## 2026-10-20, the size confound is step-count, NOT aggregation (advisor SOTA + a measurement that refines it)

The advisor's SOTA reading (FedNova "objective inconsistency": local epochs over unequal client loaders → the big client takes more SGD steps per round → it dominates the objective) predicted 2.8–6× step imbalance. **My measurement on the actual multi-survey loaders refines this: only ~1.1×.** The seismic loaders are by crossline sections, so Parihaka's larger *voxel* count does not translate to proportionally more SGD steps:

| client | survey | steps in 3 local epochs |
|---|---|---|
| 0 (A) | F3 | 210 |
| 1 (A) | F3 | 213 |
| 2 (B) | Parihaka | 237 |
| 3 (B) | Parihaka | 237 |

**Consequence:** equal-steps (the FedNova-consistent fix; I wired `max_steps` into `ClientTrainer` regardless, as the SOTA-correct control) will **NOT** fix the F3 dip — F3's 0.68 is not step-dominance. It is **task/label interference through the shared body+head** (Parihaka's 11× larger label mix shifts the shared 12-class geometry). This is consistent with FedPer already recovering F3 (0.82): local heads remove the interference. So the *remaining* problem is purely H3 (FedPerBN's per-client heads lose Parihaka's within-survey pooling), and the fix is **per-survey heads**, not equal-steps.

## 2026-10-20, per-survey heads (FedPerSurveyBN / `fedpersurvey`) — PREDICTION (written before scoring)

Setup B with the head shared WITHIN each survey (F3 sites pool rows 0–5, Parihaka sites pool rows 6–11), body shared across all 4, BN local, masked loss (no cross-survey coupling), no self-mixing. `fedseismic/federated/survey_heads.py` (`aggregate_survey_heads`: body averaged over all sites, each outc row averaged over its survey's sites only). 3 seeds running, deployed + masked present-class mIoU.

- **PREDICTION: F3 ≈ 0.82 AND Parihaka ≈ 0.78+.** Rationale: per-survey heads restore Parihaka's 2-site head pooling (the 0.672→0.80 gap is H3) and keep F3's interference removal (0.816). Both surveys should reach ~solo (0.80) or better. **This is the Setup B that should finally satisfy the go/no-go (personalization on each survey + multi-survey ≥ solo on ≥1).**
- If this holds: Chapter 4 headline = FedPer+BN with per-survey heads, and the size/task-confound story (shared-on-local N/A).
- If Parihaka STILL < 0.72: the gap is body-level (the shared body degrades Parihaka's volume features), and only a Parihaka-local body would fix it → then report the FedPerBN dip as the honest multi-survey boundary.

Implementation notes (committed `1d9783d`, 85/85 green): the deployed per-survey model suppresses untrained head rows (bias −1e9, weight 0) so an unmasked argmax picks own classes; state dicts kept on CPU in `train()` to avoid a 3× GPU model copy that OOM'd the first smoke. `max_steps` added to `ClientTrainer` (config→server→trainer) as the equal-steps control.

## 2026-10-19, multi-survey go/no-go PRELIMINARY: leaning NO-GO (1-seed FedBN)

Present-class deployed-vs-deployed (the honest comparison; ignore the local-vs-shared "gain" columns — FedAvg's shared is the broken averaged-BN reference and FedBN's is N/A). Survey A = F3, B = Parihaka.

| Survey | Solo (N=2) | Multi FedAvg | Multi FedBN |
|---|---|---|---|
| A (F3) | **0.80** (0.785/0.814/0.813) | 0.69 (0.55/0.81/0.70) | 0.68 (1 seed) |
| B (Parihaka) | **0.80** (1 seed) | 0.75 (0.75/0.70/0.80) | 0.77 (1 seed) |

**Preliminary NO-GO** — solo baselines beat multi-survey on both surveys, and solo is more stable (multi F3 wobbles 0.55–0.81 vs solo 0.79–0.81). FedBN recovers the collapsed Parihaka (0.0 → 0.77) but still does not beat Parihaka-alone (0.80). Neither condition 1 (gain on each survey) nor condition 2 (multi beats solo on ≥1) is met on current data.

**Caveats before this is a decision:** FedBN has 1 seed and solo B has 1 seed. The B gap (0.77 vs 0.80) is small enough to flip with more seeds. FedBN and solo-b need 3 seeds each for a defensible call. Flagged for Carlos — this is the go/no-go decision, not to be settled in findings.

**F3 single-survey control** (`results/tables/f3_per_method.csv`, 3 seeds): FedBN ≈ FedAvg on local mIoU (0.82/0.82 at N=3; 0.91/0.90 at N=5) — no FedBN benefit on one survey, as expected. FedBN shared-on-local is N/A (convention enforced). FedAvg+localBN-recal gives +0.02–0.03 over averaged-BN, confirming the diagnostic. GroupNorm rows are the BN-free control.

## 2026-10-19, FedBN shared-model convention (advisor ruling — PENDING CARLOS, Chapter 4)

What is FedBN's "shared model"? FedBN keeps BN running stats per-client, so there is no single global normalisation and "shared-on-local" / "official-test" are ill-defined. Advisor ruling (flagged for Carlos, not settled thesis text):

- **Use (b) for the deployed model + (c) for the shared column. Reject (a).** Averaging BN stats is exactly the mechanism the diagnostic proved failing in Setup A — a number from it measures the averaging, not FedBN. FedBN deployed = shared weights + each site's own BN = (b).
- **FedBN shared-on-local = N/A** ("FedBN has no global normalisation; shared ≡ deployed"), not a synthesized number.
- **FedBN official-test = each client's deployed model (own BN) on the official test, mean over clients**, labelled "deployed → official test". Never a synthetic averaged-BN model.
- **The existing 0.43–0.58 official range is unusable as-is** (BN source unstated, pooled with FedAvg). Recompute, split by method.
- **Go/no-go uses deployed-vs-deployed:** FedAvg-deployed vs solo (prerequisite: does collaboration help at all), FedBN-deployed vs solo (the go/no-go), FedBN-deployed vs FedAvg-deployed (does local normalisation add anything). None use FedBN's global model, so the site-0 choice can't contaminate the decision. "FedAvg deployed" = global weights + global (averaged) BN, eval mode.
- **Sanity check:** FedAvg + BN recalibrated on each site's train data (the diagnostic already validated this recovers Parihaka). If it closes the FedBN−FedAvg gap, FedBN's advantage is post-hoc BN, not training. GroupNorm is the BN-free control: an edge only under BatchNorm points to a BN artifact.

Code: `fedseismic/privacy/f3_per_method.py` (F3 per-method table with these columns). Existing F3 grid shows the trap: FedBN-BN N=5 `global_on_local` 0.67 vs FedAvg-BN 0.86 — the broken FedBN "global BN" reference.

## 2026-10-19, BN aggregation is the Setup A failure mode (diagnostic)

The multi-survey FedAvg `global_on_local ≈ 0` on Parihaka was a BatchNorm artifact, not a union-head or transfer limit. Diagnostic (existing FedAvg global model, seed 0): with averaged BN running stats in eval mode, Parihaka mIoU = 0.0003/0.0004 (per-class ≈ 0, only 0.002 at one class); after recomputing BN running stats on Parihaka's own train data — exactly what FedBN keeps local — the SAME model reaches Parihaka mIoU 0.7607 (client 2) / 0.5985 (client 3), per-class IoU 0.71–0.92. The 12-class union head learned both surveys fine; the only failure was cross-survey BatchNorm statistic averaging. FedBN (BN kept local) is therefore the fix, per Carlos's proposal. The earlier "0.75 vs 0.0 personalization gap" was a broken-reference artifact (global model scored below majority-class on F3 official test, 0.10–0.19) and is not a finding. Label-space check is clean (Parihaka train and test are both 6–11). Next: train FedBN 12-class union (1 seed) for a real go/no-go number.

## 2026-10-19, solo-baseline holdout fix (go/no-go condition 2)

The solo baselines were run with `num_clients_list: [1]` — one client on the whole survey, holdout = last 20% of the whole survey. Multi-survey gives each survey 2 sites, each holding out the last 20% of its half. So "multi-survey beats solo" compared different crosslines: not like-for-like, and condition 2 was unmeasurable. Fix: solo runs at N=2 per survey (`num_clients_list: [2]`), so `partition_noniid(crosslines, 2)` + contiguous-tail holdout are byte-identical to the multi-survey per-survey sites (verified: `partition_noniid` is deterministic contiguous blocks). Condition 2 then measures exactly "does adding the other survey help these sites".

The superseded N=1 solo runs are invalid for the go/no-go (moved to `results/runs/_superseded_solo_n1/`, outside the rescore glob, kept only as a whole-survey single-client reference). Re-running solos at N=2 is required before any go/no-go read. No numbers yet — the multi-survey FedAvg seed 42 is still training and the GPU is occupied.

## 2026-10-16, Parihaka data landed (Track 2.1)

Downloaded from Mendeley Data DOI 10.17632/gnvyh3msrj.1 (jiang zishuo; redistributes AIcrowd Seismic Facies Identification Challenge Parihaka + F3 Zenodo 10.5281/zenodo.3755060): `datasets/parihaka/parihaka_data_train.npz` (data, (1006,782,590) float32) + `parihaka_labels_train.npz` (labels, same shape int8, classes 1–6). Verified loadable; classes are 1-indexed (F3 uses 0–5), no zero/background pixels in the preview. Expected AIcrowd names: basement/other, slope mudstone A, mass transport deposit, slope mudstone B, slope valley, submarine canyon — to confirm against the challenge docs. Geometry differs from F3 ((401,701,255)): section size, depth sampling, and UNet divisibility (590 and 782 are not multiples of 16) all need a documented crop/pad choice before any joint training. Loader is generic (`InlineLoader` takes any (inline,crossline,depth) cube); needs a Parihaka branch for the 1-indexed labels and the survey offset (Setup A) or per-survey heads (Setup B).

## 2026-10-16, fixed-steps verdict (expert item 1 closed) — WITHDRAWN 2026-10-17

This entry is withdrawn: the runs (Oct 16 16:59) and their shadows (17:13) predate the committed `max_steps` trainer and shadow support (commit `89322b9`, Oct 17 21:23). The numbers below were produced by uncommitted code and the shadows trained without the step cap (method mismatch), so none of them are valid. They are kept only as a record. A clean rerun (Blood, then CIFAR) with the current commit and matched-step shadows is queued behind the running multi-survey training; "Headline 3" is suspended until it lands.

Table `results/tables/balanced_steps_howmuch.csv` (Blood, T=35/165 steps/round fixed regardless of n). Counts at the floor at BOTH budgets on all channels (t35: −0.02/+0.04/+0.04; t165: +0.00/+0.07/+0.05, intervals covering the floor). Step count does not explain the CIFAR leak — dataset does. Size sits on the floor at both budgets (t165 bias +0.024, t35 −0.262): fixing compute removes the size leak by construction, confirming its mechanism. Oracle fractions partially leak at the high budget only (t165 decaf/bias +0.14/+0.13 above floor; t35 ≈ 0): more steps refine within-support fractions without restoring counts. Support stays high (0.63–0.91); utility 0.90/0.93.

Headline 3 (suspended — see withdrawal note above; do not cite): balanced minibatches hide exact counts at FedLC-level utility on all datasets tested; dataset size leaks unless steps are fixed (then it doesn't); fractions leak partially at high step budgets; CIFAR counts leak regardless (dataset, not steps).

## 2026-10-16, balanced-minibatch on CIFAR-10 (tiebreak: Blood wins, OrganC partial, CIFAR leaks)

Table `results/tables/balanced_mb_cifar_howmuch.csv` (3 seeds). Counts R² on ALL channels: row +0.196 [+0.119,+0.191], Decaf +0.355 [+0.283,+0.369], bias +0.201 — well above the floor. Size +0.782 via bias. Oracle/estimated fractions +0.21–0.36. Support 0.91–0.99. Utility 0.880.

Dataset gradient across the three minibatch grids: Blood counts at floor → OrganC Decaf partial 0.17 → CIFAR 0.20–0.36 on every channel. It tracks local compute per round: CIFAR clients average 1670 examples (≈33 balanced batches × 5 epochs ≈ 165 steps), Blood ≈340 (≈7 × 5 = 35 steps). The more steps between the balanced ideal and the upload, the more count information returns through update magnitudes — the same mechanism as the size leak. The B-minibatch headline is now per-dataset: hides counts where clients are small, leaks where they are large. The fixed-steps-per-round variant (compute independent of n) is the direct test and goes next on the minibatch line.

## 2026-10-16, balanced-minibatch on OrganC (replication)

Table `results/tables/balanced_mb_organc_howmuch.csv` (3 seeds). Counts: Decaf +0.166 [+0.092,+0.197] (above floor), row +0.072 [+0.006,+0.073] (barely above), bias +0.068 (covers). Unlike Blood (all at floor), OrganC leaks counts partially through Decaf's positive-sum — about 17% of count variance. Size leaks harder than Blood via bias: +0.674 [+0.428,+0.788] vs +0.446. Oracle/estimated fractions: Decaf +0.150/+0.148, both above floor (Blood: oracle partial only for bias/Decaf at +0.14–0.16, estimated at floor). Support: decaf 0.92, bias 0.79, row 0.77. Utility 0.939, level with FedLC-OrganC (0.935).

Net vs Blood: same shape (support high, size leaks via bias, utility at FedLC level) but Decaf counts and estimated fractions stay above the floor on OrganC. The headline softens to "counts at or near the floor (Blood) to partial (OrganC Decaf 0.17)". CIFAR-10 minibatch runs next as the tiebreak.

## 2026-10-16, seismic shadow calibration: negative (pilot closed)

12-shadow pilot, N=5 FedAvg batchnorm seed 0, single 5-epoch updates from the round-20 received model with the victim's exact loss/optimizer. First attempt (random-boundary contiguous chunks) was misdesigned: shadows varied in size (5.5–41 Mpix) while victims are equal-size (11.5 Mpix) — the line learned size→update mapping the victims don't have. Rerun with fixed-size (112-line) Dirichlet-mix chunks: RidgeCV saturates near max penalty (α=1000 row, 30 bias), calibrated counts R² −0.004/+0.044, Spearman +0.93/+0.98 (class-identity dominated, as established). No transferable count slope: the hand-built within-class Pearson stands as uncalibrated, and no exact-count claim transfers on F3. No expansion — pilot closed per the pre-registered rule. Shadows live under `results/attacks/seismic_shadows/` (gitignored).

## 2026-10-16, buffer-holdout outcome

6/6 runs complete (N=3, FedAvg/FedBN batchnorm, buffer 10). mIoU from history: FedAvg buffer local 0.712 vs adjacent 0.723; FedBN 0.711 vs 0.721. Local-minus-global-on-local gap: +0.005/+0.056 (buffer) vs +0.004/+0.008 (adjacent). (Present-class rescore moves both arms to 0.82–0.91; the 0.01 buffer-vs-adjacent conclusion is unchanged.) The pre-registered 50% criterion is vacuous against near-zero gaps, so the honest reading: a 10-crossline buffer drops local scores by only 0.01 — holdout adjacency was NOT inflating the utility story. The deployed model genuinely matches the shared model on within-site holdouts (≈0.71–0.72 both), and both sit above official-test (0.43–0.58). Conclusion: no personalization gap on these F3 sites under either holdout; the survey-shift gap stands. The buffer flag stays available; no rerun needed.

(Pre-registered criteria, Oct-15: N=3, FedAvg/FedBN batchnorm, 3 seeds, buffer 10, same 20 rounds. Validity criteria fixed in advance: (a) the comparison is valid if all 6 runs complete with non-empty train and test splits (checked programmatically); (b) "utility preserved" means the local-minus-global-on-local mIoU gap under buffer is within 50% of the adjacent-holdout gap —otherwise the buffer changed the task, not just the holdout; (c) leakage numbers are reported for both holdouts side by side with no claim that one is "true". If (a) fails, shrink the buffer to 5 and rerun. No post-hoc threshold moves.

## 2026-10-15, Q3-from-the-sum and peer oracle (E1/E2 rescores)

Two metrics added to both sums tables (CPU rescores, warm caches). (1) `sumsum_{channel}_union_presence_auc`: same member×union-class labels as the individual union metric, scored with the shared summed vector. Within-class attribution from the sum is chance by algebra (one vector for all members); the pooled number measures prevalence. Blood α=0.1: FedAvg 0.51–0.58 (chance to weak prevalence), FedLC row 0.62–0.67, decaf 0.59–0.62. The sum does not attribute classes to members — it preserves, at best, which classes are widespread. That is the number supporting "the sum reveals only joint absence". (2) `peer_{channel}_prop_oracle_r2`: line prediction on the peer residual renormalized on the true union of the others' supports, vs the others' mean proportions. FedAvg: +0.48–0.62 at N=2, +0.24–0.39 at N=3, Decaf +0.24 at N=5, dead at N=8. FedLC: +0.29–0.46 at N=2, row +0.15 (covers 0) at N=3, dead beyond. Told the support, peers read fractions only in tiny cohorts. N=2 peer==individual verified to 6 decimals on counts (algebra sanity).

## 2026-10-14, member-level support conditioned on the union (expert Q3)

New metric `individual_{channel}_union_presence_auc` in both sums tables: presence AUC restricted to classes held somewhere in the cohort, so always-absent classes (free negatives) stop inflating the score and what remains is attribution — pinning k on member j. Blood α=0.1: FedAvg row 0.93 / 0.85 / 0.80 / 0.72 at N=2/3/5/8 (attribution degrades as cohorts grow and share each union class), bias 0.89–0.92 flat, decaf ≈1.00. FedLC row 0.98–0.99 at every N (row magnitudes separate members at any cohort size), decaf 0.97–0.99, bias at chance (0.37–0.50, the logit-adjustment signature again). Third panel added to `leakage_vs_silos.png`. The Q3 reframing lands: union detection (peer/sum presence) is easy for everyone; member attribution is where FedLC pulls away from FedAvg with N.

## 2026-10-14, recalibrated attackers (expert issue A)

All count lines refit as floor-plus-shrunk-residual: shadow per-class mean plus a RidgeCV penalty on the standardized single-channel score (`howmuch._FloorLine`, alphas 10⁻⁶–10³, LOO on shadows). With no signal the slope shrinks to zero and the prediction lands on the floor (R²=0 by construction of the per-class-mean baseline). Size fits got the same treatment (`_fit_floor_ridge`). New calibration-free measure: pooled Spearman of predicted vs true counts, reported against a population-profile floor (big classes rank high for everyone: Spearman ≈ +0.14 on the N=2 check).

Re-scored with warm shadow caches: `sums_blood_a01/a05.csv`, all three `balanced_*_howmuch.csv`, `howmuch.csv` (prior-smoothing grid). Substantive results do not move: server-sum counts stay at the floor (recalibrated N=8 FedLC row −0.023 instead of −0.05); balanced counts hug the floor (worst −0.04, no more −0.15 depths). Phrasing rule adopted: R² intervals covering the floor read "not distinguishable from the floor", never "below the floor".

The Spearman column initially looked like a new leak (balanced bias +0.69–0.77, Decaf +0.46–0.76). Decomposition check on Blood balanced-1: all-pairs Spearman +0.75, present-only Spearman −0.07, present-only Pearson −0.17. The rank signal is pure support detection (absent≈0 vs present≠0), not count reading within support. Oracle proportions R² ≈ 0 says the same thing from the other side. Spearman therefore redundantly confirms support; it adds no count-leakage evidence under balancing. Unit tests added (`test_line_falls_back_to_the_floor_without_signal`, `test_line_recovers_a_real_slope`); plain least squares scores −0.27 on the same seed.

## 2026-10-14, workstream G prediction

Written before any seismic attack runs (one FedAvg timing run in progress, no scores). Setup: F3 Netherlands, contiguous crossline sites, N=3 (233 lines each) and N=5 (140 each), full participation, 20 rounds. UNet with BatchNorm (FedAvg uploads running stats; FedBN keeps them local) and GroupNorm (no running stats — the control). Per-site facies pixel counts are the "how much": c4 ranges 0.01–0.07 across sites, c5 0.00–0.04 (absent in sites 0–2 at N=3), c3 0.04–0.09. Shadows from Parihaka if available, else F3 cross-survey splits — cross-survey shadows are the honest choice and are recorded as such.

Prediction. The outc-head row-size channel should read per-site facies counts under FedAvg the way the classifier rows read class counts: counts R² above the floor, with the FedAvg control gating. FedBN should match FedAvg on the head channels (it only privatizes the BN stats) while the BN running-mean channel goes dark — the leakage gap between the BatchNorm UNet and the GroupNorm UNet on the stats attack is the statistics channel, predicted positive for FedAvg and zero for FedBN/GroupNorm. Support for the rare facies (c5): presence AUC above floor under FedAvg. Local segmentation heads (personal outc) are FedPer-analogous and stay not-measured on the head. Utility: FedBN ≥ FedAvg on local mIoU (BN stats fit the site), GroupNorm close behind.

Note a bug found during setup: `extract_last_linear` picked the auxiliary `reconstruct` head (Conv2d 64→1) instead of the segmentation head `outc` (Conv2d 64→6) for UNet states, so logged head deltas tracked reconstruction, not facies. Fixed to prefer `outc.weight` when present; classification models have no `outc` key and are unaffected. The timing run used the old code and is utility-only (no `received/` saved).

## 2026-10-14, seismic holdout fix (expert issue C, code part)

`split_client_local_test` takes a `buffer` (seismic/rng=None path only): that many crosslines between train and test are dropped from both, so the holdout is spatially separated instead of adjacent. Wired as `RunConfig.spatial_holdout_buffer` (default 0 = legacy splits byte-identical, verified: 186/47 adjacent vs 176/47 with gap 10 on a 233-line site) plus sweep passthrough. The running 24-run grid keeps buffer 0; a buffer-holdout utility comparison runs after it, before any seismic number is interpreted.

Contamination check for the reconstruct-vs-outc fix: classification states have no `outc.weight` key, so every CNN number (all tables) is bit-identical before/after. On seismic, only the timing run used the old code and it saved no `received/` and feeds no attack table; all 8 finished grid runs postdate the fix; no seismic attack table exists yet. Nothing reported drew on reconstruct-head deltas.

## 2026-10-15, seismic first read (hand-built, last round only) — revised

Table `results/tables/seismic_f3.csv`. Revised after external review: (1) pooled count Pearsons are dominated by class identity, so the table now reports within-class means (each facies centered across sites); (2) presence pair counts reported (48+/6− at N=3, 81+/9− at N=5 — every negative is c5); (3) the BN metric replaced with centered per-(round, layer) z-scored AUC, overall plus early/late/first-encoder splits. Still no shadows and no CIs — exploratory, last round only.

Counts: within-class bias Pearson +0.36 to +0.64 on all 8 jobs (5–6 of 6 facies positive per job; c2, the dominant facies, is the frequent exception). The site-level count signal survives centering — more of facies k at a site goes with a larger bias move for k. Within-class row Pearson is consistently negative (−0.25 to −0.46): the inversion survives too. Presence: bias 0.90–0.99, row 0.00–0.06 (anti-reads; flipped ≈1.0).

BN running means identify sites at AUC 0.82–0.89 overall, 0.79–0.87 early, 0.93–0.97 late, 0.92–0.97 first encoder layer — late above early (statistics diverge as models personalize). FedBN ≈ FedAvg everywhere: both uploads carry BN keys in this logging (FedBN keeps them out of the aggregate, not out of the upload — a strict deployment strips them, so treat the FedBN number as a loose implementation's leak). Two caveats stand: site linkability is not facies attribution (contiguous sites differ in amplitude/depth too), and unmixing means into (π, μ) pairs is underdetermined for N<6 without shadow-supplied class-conditional means.

What stands: bias reads presence (last round, 6–9 c5 negative pairs) and within-class counts; rows invert readably; BN stats link sites increasingly over training. Downgraded: any exact-count claim (no shadow calibration), FedBN=FedAvg as a finding (expected — the head is still averaged), anything about leakage trajectories. The 0.98 local mIoU is withdrawn entirely (see pixel-accuracy bug entry below — it was never a holdout-adjacency question).

## 2026-10-15, base rates for the seismic presence claims (advisor check)

89–90% of site-facies pairs are present, so "predict everything present" already scores 0.89–0.90 accuracy. Against that base rate: bias balanced accuracy (midpoint threshold) is 0.76 at N=3 and 0.90 at N=5 — above chance (0.5) but well below what the 0.90–0.99 AUC suggests. BN top-1 nearest-neighbor match across rounds is 0.98 vs 0.33 chance at N=3 and 0.94 vs 0.20 at N=5: site re-identification from running stats is near-perfect, a much stronger statement than the AUC. Both framings now reported; the AUC-only gloss is retired.

## 2026-10-15, the 0.98 was pixel accuracy, not mIoU (user catch)

`evaluate_personalized_local` scored every client with `_accuracy` (example accuracy), which on segmentation equals pixel accuracy. On imbalanced facies that reads ~0.98 while macro mIoU sits at 0.72–0.75 (recomputed from saved states on N=3 batchnorm: train 0.82–0.95, holdout 0.66–0.81, c5 IoU 0.0 everywhere). Per-round history always held mIoU (0.73 at round 20); only the end-of-run `results.json`/`run_meta.json` local fields carry the wrong metric, because `server.run()` overwrites them from the end scorer. Fixed in code (task metric unless a log prior is applied; classification behavior identical), regression test added (`test_personalized_metric`: majority model scores <0.7 mIoU), `seismic_f3.csv` local column recomputed from history. Corrected utility (present-class mIoU re-evaluated from saved weights): local 0.82–0.91, global-on-local 0.80–0.87, global (official test, all facies present so the fix changes nothing) 0.43–0.58. The local-vs-shared gap on holdouts is ≈0.00–0.04 — no personalization gap at all under the corrected metric (the earlier 0.72–0.75 used the all-class mean). The only gap is official-test survey shift. Buffer comparison stands as run (its numbers use the same history metric on both arms, so the 0.01 conclusion is unaffected).

## 2026-10-14, seismic timing run

Single FedAvg run, N=5 contiguous sites, 20 rounds, 61 min on the RTX 3090 (~3 min/round). Final global mIoU 0.55, global-on-local 0.70. (The deployed-local 0.98 first reported here was pixel accuracy, not mIoU — see the Oct-15 bug entry; corrected local mIoU is ~0.73.) View artifacts confirm BN running stats are captured per upload (`bn::*.running_mean/var`, needed for the statistics channel). Full grid launched: `configs/sweeps/seismic_f3.json`, 8 jobs (FedAvg/FedBN × N=3,5 × batch/group norm) × 3 seeds = 24 runs, ~20h. FedBN+group is the near-duplicate control of FedAvg+group (no BN to keep local), kept as a sanity check.

## 2026-10-29, THE ROTATION BUG + reframe (advisor review; DISCLOSE in thesis)

**Disclosed finding:** Parihaka was being fed to the model rotated 90° from its natural
orientation (reflectors vertical instead of horizontal; horizontality 2.0 vs 0.5). F3 and
Penobscot were correct; only Parihaka was affected. Because every multi-survey run shares a
body with Parihaka, this touched all multi-survey numbers (the F3 +0.02, the Penobscot −0.38,
the Parihaka −0.07, and the BN blackout which used the multi-survey FedAvg run). The single-
survey privacy probes (F5 fingerprint, F6 x-ray, F7 pull) used F3-only runs and are UNAFFECTED.

**Fix:** built `parihaka_*_train_rot90.npz` (cube transposed so the loader's sections come out
with horizontal reflectors; verified horizontality 2.003 → 0.499). Retraining Parihaka solo +
 every multi-survey run on the corrected data. **All numbers reported after 2026-10-29 are
post-fix. State this in the thesis limitations: a rotation bug was found and fixed, and every
reported result is post-fix.** (Disclosing it is rigour; a committee member finding it would
sink the defense.)

**Central claim (reframed — it is a BOUNDARY/LIMIT finding, not a method win):**
> With disjoint label spaces, the only federated setup that keeps solo-level accuracy across
> surveys keeps the head per-survey and BatchNorm local. Cross-survey body sharing gives no
> reliable gain and can collapse a distinct survey. The parameters personalization has to share
> (head deltas within a survey) leak which facies a site has, but not how much.

fedpersurvey is the EVIDENCE behind the boundary, not the contribution.

**Threat-model fix (BN re-identification):** the 0.989 site re-ID from BN running stats only
applies when BatchNorm is SHARED (FedAvg-style). In FedBN / fedpersurvey, BN is kept local, so
the server never sees the running stats — that leak is closed by personalization. This is a
better story: local BN improves accuracy AND removes a privacy leak at the same time. Make it a
central figure.

**Drop / reword:** "cross-survey helps the small survey (F3 +0.02)" is WITHIN the F3 solo seed
spread (0.771–0.804, spread 0.033 > effect 0.02). Report as "no significant gain" — do not
claim F3 is helped. The defensible asymmetry is the HARM to large/distinct surveys (Penobscot
collapse, Parihaka dip), not a small-survey benefit.

**"Absence is loud" (reframe for sponsors):** the facies a site LACKS leaves the LARGEST head
bias change (false positives get corrected hard). So "a site lacking the Zechstein is the
easiest to spot." Practical risk framing for operators.

## 2026-10-29, final-24h experiment plan (advisor priorities)
- P0 (GPU queue running): retrain on corrected Parihaka — solo_PARIHAKA (done), multi
  fedpersurvey 2-survey (running), multi fedpersurvey 3-survey (asymmetry), FedBN + FedPerBN
  2-survey (baselines).
- P0: re-run the BN blackout on the corrected multi checkpoint (the F3 figure used the old
  rotated run). Privacy probes F5/F6/F7 are F3-only → no re-run needed.
- P1 (inference, cheap): all-class re-score of everything; buffered re-score (drop the N
  test crosslines next to the train boundary) to quantify the adjacency inflation.
- P1: 2-3 seeds for the Penobscot solo-vs-multi collapse (rare-class collapse can be seed-dependent).
- P2: F3 solo extra seed (settles "small survey helped" = noise).
- P3: centralized F3 on the official Alaudah split (a benchmark anchor interpreters recognise).

## 2026-10-29, limitations to state explicitly (advisor)
- Three surveys, one architecture, contiguous-block sites (no realistic between-operator shift).
- Seed counts per result; mark single-seed results clearly.
- Present-class vs all-class and buffered vs unbuffered inflation numbers.
- Absolute mIoU is not benchmark-comparable (except the P3 anchor if done).
- Threat model: honest-but-curious server observing updates; no DP / secure aggregation evaluated.
- The leak attacks are simple (bias deltas, BN matching) → the leaks measured are LOWER bounds.
- Disclose the rotation bug + that all results are post-fix.

## 2026-10-29, ROTATION RETRAIN RESULT — the "bug hurt performance" hypothesis is WRONG

Solo Parihaka (K=6, 2 clients, 20 rounds, 1 seed), ONLY the data orientation differs:
- original input (reflectors VERTICAL, horizontality 2.00): present-class **0.845**
- corrected input (reflectors HORIZONTAL, horizontality 0.50): present-class **0.7015**

**The rotation does NOT improve Parihaka — it is ~0.14 WORSE.** The model learned the original
(vertical-reflectors) input perfectly well (0.845). So the premise "the input is rotated 90° and
that hurt performance" is NOT supported: the orientation did not cost accuracy. If anything the
original orientation was *inflating* the number.

**Interpretation (be careful, CORRECTED):** the orientation DOES change accuracy (0.845 vs
0.70 = 0.14), and the ORIGINAL (vertical) orientation is the BETTER one. So the premise that the
rotation *hurt* performance is backwards: the non-natural orientation was, if anything, *helping*
(likely a UNet/section-geometry preference). The correct statement is: "the input orientation
changes accuracy by ~0.14, and the model does BETTER on the original orientation; therefore the
'rotation bug' did not cost us accuracy, and the multi-survey results were not hurt by it."
The figures should still be shown in the natural (horizontal) orientation for interpreters — that
is a DISPLAY choice — but we must NOT claim the rotation fixed an accuracy problem.

**Subtle caveat (rigour):** my training cube was built with `transpose(2,1,0)`, which produces
sections equal to `np.rot90(display_section)` FLIPPED along the inline axis (verified:
`disp == transp[:, ::-1]`). Both are horizontal (0.499) so the ORIENTATION effect above is robust,
but the absolute 0.7015 may shift by ~±0.02 under the exact display orientation. The RELATIVE
claims (multi vs solo, same rotation on both sides) are unaffected by the flip.

**Recommendation for the thesis:** (a) keep reporting the ORIGINAL numbers (they were not hurt by
orientation) OR (b) report the corrected-data numbers (0.7015) as the honest orientation — but do
NOT claim the rotation *improved* anything. Either way, disclose that the input orientation was
found to be non-natural, that it did not affect model accuracy, and that figures are shown in the
natural orientation. The safest framing: "we verified the input orientation does not change model
accuracy (0.845 vs 0.70 across orientations, 1 seed); all figures are displayed in the natural
geological orientation."

## 2026-10-29, honest re-score: quantifying the inflations (advisor P1)

`talk_figs/rescore_honest.py` re-scores checkpoints dropping K test crosslines next to the
train boundary (buffer) and reporting all-class alongside present-class. Parihaka solo
(un-rotated, site 0), present-class = all-class (all 6 classes present at this site, so no
present-class inflation here — that inflation only bites at sites with an absent facies):

| buffer K (test xlines dropped) | present-class | all-class |
|---|---|---|
| 0 (contiguous holdout) | 0.839 | 0.839 |
| 5 | 0.812 | 0.812 |
| 10 | 0.796 | 0.796 |

**Adjacency inflation ≈ 0.027 (K=5) to 0.044 (K=10).** So the contiguous train/test holdout
inflates Parihaka solo by ~0.03-0.04; a buffer of ~10 test crosslines removes it. This is the
number for limitation (b). The corrected (rotated) solo is being re-scored the same way.

Present-class vs all-class inflation is 0 at this site (no absent facies) — it only inflates at
sites that lack a class (e.g. F3 client 0 lacks Zechstein -> 0.79 all-class vs 0.95 present-class).
Report both metrics; the inflation is site-dependent, not global.

## 2026-10-29, ROTATION COMPARISON: BUFFER CONFOUND

**Correction:** The notes immediately below overstated the finding. The metadata
shows a buffer and orientation confound; it does not determine either one's
contribution to the difference. A UNet's function class can be transposed by
transposing its kernels, but a fixed learned network is not rotation-invariant,
and mixed-orientation federated training is not a global transpose. The isolated
effect is unknown without matched controls. Do not cite the following historical
interpretation as established.

The 0.14 Parihaka drop (0.845 original -> 0.7015 "rotated") is NOT a rotation effect. The
config diff between the two solo runs shows they used DIFFERENT train/test splits:
`spatial_holdout_buffer: orig=0 vs rot=50`. The rotated config (seismic_multisurvey_rot.json)
inherited `spatial_holdout_buffer: 50` (the de-leak setting added earlier to
seismic_multisurvey.json), while the original used 0. A 50-crossline gap between train and test
= less training data + a harder (non-adjacent) test -> lower mIoU. That is the ~0.14, not the
orientation.

This CONFIRMS the advisor's symmetry argument: a UNet is transpose-symmetric (conv/BN/pool
isotropic), so rotating the input axes cannot change accuracy by 0.14. The rotation is a
symmetry; the buffer is the real effect.

**Consequences:**
1. The "rotation hurt performance" hypothesis is WRONG. The rotation does not change accuracy.
2. The rotation-retrain numbers (solo 0.7015, multi2 0.724/0.719) are actually the **buffer=50
   (de-leaked)** numbers, not "corrected-rotation" numbers. They ARE the honest, non-adjacent
   numbers the advisor's P1 wanted — so the retrain is still useful, just for a different reason.
3. To isolate the rotation, re-run with buffer=0 (expected ~0.845, matching the original).
   To isolate the buffer, compare buffer 0 vs 50 at the same orientation (expected ~0.05-0.14).

**Reframe:** the retraining (all runs at buffer=50) is the DE-LEAKED evaluation (the honest,
non-adjacent split), NOT a rotation test. The rotation itself is display-only (the figures show
the natural orientation; training is orientation-agnostic). The ~0.05 adjacency inflation measured
earlier (buffer 5-15) is consistent with this: the buffer removes the train/test adjacency.

## 2026-10-29, DOES THE MIXED-ORIENTATION SHARED BODY SPLIT? (probes on a trained multi-survey ckpt)

**Correction:** This exploratory probe used the same client-0 personal model for
both surveys and did not compare Parihaka's two orientations. High cosine of
average absolute activations, and the filter-pair null used, cannot establish
shared representations or determine the effect on accuracy. The verdict below
is withdrawn; orientation as a contributor to interference remains untested.

User's sharp question: we train the shared body on F3 (natural, reflectors horizontal) + Parihaka
(rotated 90deg, reflectors vertical) = a PARTIAL transpose. Does that have a consequence for the
conv layers even though a GLOBAL transpose is a symmetry? Advisor: the reasoning is sound (a
partial transpose is not a symmetry -> oriented filters duplicate -> capacity cost), but the effect
is probably SMALL (little sharing to lose between two different geologies; local BN routes
channels cheaply; wide UNet has spare width). Ran the advisor's no-retrain probes on the trained
multisurvey_fedpersurvey checkpoint:

PROBE (a) per-channel activation profile, F3 vs Parihaka (cosine / one-survey-channel frac at 2x):
  inc.double_conv.0            cosine=1.000  one-survey=0.02
  inc.double_conv.3            cosine=0.968  one-survey=0.70
  down1..double_conv.0         cosine=0.978  one-survey=0.11
  down1..double_conv.3         cosine=0.965  one-survey=0.24
  down2..double_conv.0         cosine=0.941  one-survey=0.24
  down2..double_conv.3         cosine=0.948  one-survey=0.17
  -> The activation profiles are NEARLY IDENTICAL (cosine 0.94-1.00). The shared body uses the
     SAME channels for F3 and Parihaka. It is NOT split by orientation. (conv2 shows mild channel
     specialization: 70% of channels differ >2x in magnitude, but the profile shape is the same.)

PROBE (b) transpose-pair duplication (conv1..conv4): only conv1 has excess transpose pairs
  (frac cos(W_i^T,W_j)>0.8 = 0.281 vs 0.156 null); conv2+ have ZERO. conv1's oriented edge-filters
  are normal for any conv net, so this is shallow/weak evidence, not a deep orientation split.

VERDICT: the mixed orientation is essentially ABSORBED by the isotropic UNet. The shared body is
not split by orientation (high activation cosine + no deep transpose duplication). So "no
consequence for the conv layers" is essentially CORRECT -- the multi-survey interference comes from
the GEOLOGY (different facies), not the orientation. The advisor's "small effect" is confirmed.
A partial transpose is theoretically not a symmetry, but its practical cost here is minor. The
rotation remains a data-prep bug worth disclosing (reflectors display vertical), but it is not a
material driver of accuracy or cross-survey interference. Kill criteria largely met (few deep split
pairs + high activation cosine), so do NOT spend GPU re-testing orientation.

## 2026-10-29, LOSS AUDIT (advisor): every loss — principled or "because we did"?

**Correction:** This preliminary audit did not verify all trainer overrides.
`FedPerSurveyBNClientTrainer.train` uses masked cross-entropy, including in the
matched-size experiments; do not describe all seismic FL as FocalDice. Check
the trainer for each reported run. Dice and IoU are related but distinct
objectives; class-weighting choices require empirical justification, not a
claim that adding weights necessarily double-counts imbalance. There is no
evidence here that hyperparameters were fixed before experiments; do not claim
that without a dated record. Named-method comparisons below require independent
paper/code verification before being cited as established.

SEGMENTATION — FocalDice(1:1, gamma=2, alpha=None) for the seismic runs: PRINCIPLED, keep it.
- Compound Dice+Focal is the most robust loss family (Ma et al. 2021 "Loss Odyssey", MedIA); MONAI's
  DiceFocalLoss default is the same 1:1. Per-class mean Dice targets per-class mIoU (our metric).
- alpha=None is DEFENSIBLE: macro-Dice already weights classes equally; a single global alpha is
  ill-defined under non-IID label skew; a per-site alpha would add a personalization confound.
  (Do NOT claim RetinaNet's alpha=0.25/gamma=2 pairing for multiclass — that's binary detection.)
- gamma=2 = Lin et al. 2017 default. NOT tuning is OK if stated: "settings fixed before experiments,
  library defaults, identical across all FL methods; our claims are method comparisons, not absolute
  scores." A small centralized sensitivity table (gamma in {0,2}, Dice weight in {0.5,1,2}) closes this.
- No Lovasz/IoU loss is DEFENSIBLE: Eelbode et al. 2020 (IEEE TMI) show soft Dice ~ soft Jaccard for IoU
  metrics. Cite Eelbode; mention Lovasz-Softmax (Berman 2018) as future work.
- **LANDMINE (must state):** with smooth=1, Dice heavily penalizes predicting an ABSENT class -> the
  model learns never to predict it -> the vacant-class drift that FedVLS/FedSeis then fix. So FocalDice
  partly CREATES the problem those methods solve. Say this explicitly (and state Dice is per-image vs
  per-batch).

CLASSIFICATION — CE vs LA: principled, but one CONFOUND.
- LA with tau=1 IS Balanced Softmax (Ren 2020) — don't present them as two alternatives.
- Menon 2021 removes log pi at inference; we ADD it back -> local-prior re-injection. Name it that
  ("logit-adjusted training with local prior re-injection"); it's a personalization choice.
- **CONFOUND:** LA trains the shared weights to be prior-independent (moves the prior into local log pi).
  That mechanism ALONE can hide the class mix, so a leak drop under LA may come from the LOSS, not the FL
  scheme. Add a class-weighted-CE control (fixes imbalance WITHOUT separating the prior) or reword the
  finding as a property of the loss.

METHOD NAME-vs-IMPLEMENTATION (same class of issue as FedLC):
- FedPer (Arivazhagan 2019): MATCHES. | FedBABU (Oh 2022): MATCHES (state: fine-tune head only or full).
- **FedRoD (Chen & Chao 2022): MISMATCH.** Paper applies Balanced Softmax to the GENERIC head alone; the
  personalized g(z)+p(z) uses plain CE; the personal head's gradient does not reach the body. We add
  log pi to the SUM -> NOT the paper's method. Rename "FedRoD-variant" or fix the code.
- **FedVLS (Guo 2025): MISMATCH.** The paper's 2nd term is logit SUPPRESSION (penalize non-label logits),
  not CE(z + log freq) (=LA). + Dice makes it a segmentation adaptation. Call it "FedVLS-inspired
  (LA + vacant distillation)".
- **FedKPer:** confirm whether it's published; if it's ours, say so. Its KD is KL(student||teacher) =
  REVERSE KL (matches teacher's top classes only); standard KD (Hinton 2015) is KL(teacher||student) with
  T^2 scaling. Justify the direction or flip it. lambda=10 needs a sweep.
- **FedSeis:** ours. Cite FedProto (Tan 2022) for the prototype alignment.

RANKED CHAPTER-3 JUSTIFICATION (most->least solid):
1. FedPer/FedBABU + CE: identical to the source papers.
2. FocalDice 1:1, gamma=2: compound-loss robustness (Ma 2021), MONAI default, macro-Dice -> mIoU (Eelbode 2020).
3. alpha=None: macro-Dice already balances; global alpha ill-defined under non-IID.
4. LA (tau=1): Menon 2021 / Ren 2020; pi re-injected for local deployment.
5. FedVLS/FedSeis: adaptations, labelled as such.
6. FedKPer/FedRoD: FIX implementation or RENAME first.

COMMITTEE QUESTIONS TO PRE-EMPT:
Q "You optimize Dice but report IoU, and didn't tune." -> cite Eelbode 2020, state fixed-in-advance, show the small sensitivity table.
Q "Is the leak reduction the defense or LA?" -> add class-weighted-CE control, or reword as a property of the loss.
Q "Is this really FedRoD/FedVLS?" -> rename or fix before submission.

## 2026-10-30, SIZE-MATCHED PARITY EXPERIMENT (2-client FL vs solo, 350 sections/survey, buffer=50, batch=2)

Setup: 2-client FL [F3-350 + Parihaka-350] vs each survey trained ALONE at 350 sections, same split,
same 12-class union head, masked cross-entropy loss on training-present classes, 2 seeds (0,1).
Readout: Delta_s = mIoU(FL,s) - mIoU(solo,s), per survey (present-class mIoU,
pixel-pooled within each client, argmax restricted to its survey's six rows).
The +/-1.5-point margin was proposed before these scores were inspected; it is
not by itself an equivalence test. Training sections rather than pixel counts
were matched, and Parihaka sections contain substantially more pixels.

AGGREGATE present-class mIoU (A=F3 classes 0-5, B=Parihaka classes 6-11):
  FL        seed0: F3 0.536   Pari 0.408
  FL        seed1: F3 0.498   Pari 0.396
  F3 solo   seed0: F3 0.516   |  seed1: F3 0.399
  Pari solo seed0: Pari 0.452 |  seed1: Pari 0.355

DELTA_s (FL - solo), per survey per seed:
  F3:   seed0 +0.02048; seed1 +0.09915 (mean +0.05981).
  Pari: seed0 -0.04429; seed1 +0.04110 (mean -0.00159).

**Interpretation:** The observed seed-0 effects have opposite signs; the F3
difference is positive in both seeds, while the Parihaka difference changes
sign. Results are descriptive, not proof of parity, non-parity, consistent
asymmetry, a mechanism, or the absence of a data-volume effect. F3 solo's
0.117 spread illustrates instability, but it is not an estimate of the
uncertainty in paired differences. Only two seeds are planned/completed.

**Important correction:** F3 class 5 has **zero** pixels in the actual training
sections 0–349, but 534,099 pixels in test sections 561–700. Its test IoU is 0
in all F3 runs because the trainer masks the loss to classes observed in
training. The logged `labels/pi.json` instead describes the larger pre-subset
pool and incorrectly lists class 5 as present. This metadata mismatch also
affects class-support information passed to the aggregator.

The requested 50-crossline buffer is a minimum: selecting the first 350
sections leaves 211 crosslines between F3 train and test and 276 for Parihaka.
Train and test inputs were min-max normalized using the entire survey cube,
including the holdout; the preprocessing is matched between each survey's arms
but uses unlabeled test data. Parihaka has 593,540 pixels per section versus
F3's 102,255 (5.80×), so the data match is in section count, not pixels.

Seed-0 FL `history.jsonl` has stale rounds 1–19 followed by the completed
batch-2 run's rounds 1–20; do not use the unfiltered history for learning
curves. Scores above were computed from the saved final personal checkpoints,
not from that history. All six runs recorded a dirty Git tree, so the exact
working trees cannot be reconstructed solely from their recorded commits.

PER-CLASS IoU (the survey's own 6 classes; Parihaka classes shown 6-11 in the union):
  F3 (classes 0-5), seed0:  FL [0.955,0.751,0.922,0.457,0.133,0.000]  solo [0.935,0.715,0.932,0.346,0.169,0.000]
     Delta:                      [+0.020,+0.036,-0.010,+0.111,-0.036, 0.000]
  F3 (classes 0-5), seed1:  FL [0.942,0.764,0.908,0.264,0.111,0.000]  solo [0.811,0.535,0.799,0.188,0.062,0.000]
     Delta:                      [+0.131,+0.229,+0.109,+0.076,+0.049, 0.000]
  Parihaka (classes 6-11), seed0: FL [0.842,0.496,0.303,0.524,0.197,0.086]  solo [0.897,0.550,0.421,0.596,0.096,0.152]
     Delta:                          [-0.055,-0.054,-0.118,-0.072,+0.101,-0.066]
  Parihaka (classes 6-11), seed1: FL [0.719,0.502,0.447,0.509,0.032,0.167]  solo [0.825,0.433,0.231,0.293,0.196,0.152]
     Delta:                          [-0.106,+0.069,+0.216,+0.216,-0.164,+0.015]
  -> These per-class differences describe the runs, not a causal explanation.

Advisor review (30 Oct): the matched-size result is inconclusive with two seeds;
neither parity nor consistent harm to Parihaka is established. Five seeds would
be more informative, but do not launch more without the requested decision.
The three-survey seed-0 pilot has priority. Parihaka was in the original training
orientation, and section counts, not pixel counts, were matched.
