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

## 2026-10-15, seismic first read (hand-built, last round only)

Table `results/tables/seismic_f3.csv`. All scores hand-built (Pearson on log pixel counts, presence AUC); no shadows, no CIs — 3 seeds × 3–5 sites is too thin for intervals, and UNet shadows (48 × 5 epochs each) are queued only if this control passes. Outc deltas are round 20 only (rounds 1–19 unrecoverable, see head-fix entry).

Presence reads through the bias at 0.91–0.99 AUC on every job (FedAvg and FedBN, batch and group norm). The row-size channel ANTI-reads presence (AUC 0.00–0.06: absent classes move more). Per-site check on N=3 seed 0: c5-absent sites have c5 row sizes 0.37/0.32 vs 0.13 where present, and c5 bias −0.046/−0.006 vs +0.016. Mechanism: with no c5 ground truth, every c5 prediction is an error, so the c5 row accumulates large corrective gradients exactly where the facies is missing. A flipped sign still counts as readable, same rule as the classification defense.

Counts: bias–log-count Pearson +0.55 to +0.78 on all 8 jobs; row–log-count Pearson −0.52 to −0.86 (same inversion — at convergence the fitted majority facies barely move). FedBN matches FedAvg on every head number, as predicted (it only privatizes BN stats).

BN statistics do NOT identify sites here: same-site vs different-site cosine AUC on running-mean stacks is 0.56–0.57 for FedAvg and 0.57 for FedBN — no gap, both near chance. The predicted statistics channel (FedAvg leaks, FedBN/GroupNorm dark) is not supported by this metric. Caveats that could hide it: all 20 rounds pooled (early rounds may separate better), raw stacked means dominated by large layers, cosine on unnormalized stacks. A per-layer early-round cut is the next diagnostic before calling this negative. Note the uploads DO carry BN keys under both methods (FedBN keeps them out of the aggregate but not out of the logged upload); a strict deployment would strip them.

Utility: local mIoU 0.977–0.982 everywhere (adjacent holdout — inflated per issue C, buffer comparison still queued); global mIoU 0.53–0.58 except FedBN N=5 batch at 0.43 (local stats fragment the shared model, expected). The personalization gap is real even if its size is inflated.

Prediction vs outcome: outc reads counts — holds for bias, inverted-but-readable for rows. FedBN≈FedAvg on head — holds. BN-stats gap — does not hold on this metric (diagnostic queued, not a verdict). c5 presence — holds via bias.

## 2026-10-14, seismic timing run

Single FedAvg run, N=5 contiguous sites, 20 rounds, 61 min on the RTX 3090 (~3 min/round). Final global mIoU 0.55, deployed-local mIoU 0.98, global-on-local 0.70. The personalization gap is enormous — sites differ enough that the shared model loses nearly half its mIoU off-site. View artifacts confirm BN running stats are captured per upload (`bn::*.running_mean/var`, needed for the statistics channel). Full grid launched: `configs/sweeps/seismic_f3.json`, 8 jobs (FedAvg/FedBN × N=3,5 × batch/group norm) × 3 seeds = 24 runs, ~20h. FedBN+group is the near-duplicate control of FedAvg+group (no BN to keep local), kept as a sanity check.
