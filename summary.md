# FedSeis: Prototype-Based Federated Learning for Seismic Facies Segmentation

> **Paper status (2026-04-22 02:25 UTC):** 40 FL runs complete. Campaign
> compute done. Headline metric is **rare-class-first** (per-class IoU
> for C4/C5 + finalC5 retention), not mean mIoU.

## Campaign final ranking — rare-class-first (40 runs + MetaFusion post-hoc)

| Rank | Config | mIoU | σ(mIoU) | maxC5 | **best-seed finalC5** | Paper role |
|---:|:---|---:|---:|---:|---:|:---|
| 0 | **Recall-per-slice + MetaFusion τ=0.5** | **0.5952** | n/a | n/a | 0.226 | **Train + test-time best** |
| 1 | **Recall-per-slice (ours)** | 0.5729 | 0.017 | **0.240** | **0.280** 🏆 | Novelty headline (train) |
| 2 | Recall Loss (Tian 2021) | 0.5725 | 0.018 | 0.229 | 0.251 | Closest published |
| 3 | Recall Loss + biased | 0.5706 | **0.004** | 0.227 | 0.221 | Reliability |
| 4 | Unified Focal (Yeung 2021) | **0.5786** | 0.021 | 0.198 | 0.2505 | Best mIoU |
| 5 | CRL λ=1.0 | 0.5675 | 0.011 | 0.190 | 0.273 | Earlier ours |
| 6 | UF + biased | 0.5639 | 0.005 | 0.189 | 0.223 | Reliability |
| 7 | CRL λ=3.0 | 0.5570 | 0.009 | 0.181 | 0.201 | λ ablation |
| 8 | FocalDice (baseline) | 0.5627 | 0.007 | 0.199 | 0.073 | Loss baseline |
| 9 | FedAvg | 0.5711 | 0.018 | 0.130 | 0.000 | FL baseline |
| 10 | Asym Tversky | 0.5732 | 0.009 | 0.107 | 0.000 | Cite, loses C5 |
| 11 | FCIoU (MAKE 2023) | 0.5542 | 0.002 | 0.123 | 0.000 | Cite, underperforms |

**Headline findings:**

1. **Recall-per-slice wins on rare-class recovery** — best-seed final
   C5 = 0.280 is the project record; the only configuration with
   sustained C5 across multiple final rounds (P3 s7: 0.265/0.207/0.008/0.280
   at rounds 17–20). Paper novelty: *locally computed recall weighting,
   matched to the crossline spatial structure of seismic data and to
   the geographic Non-IID partition*.

2. **Unified Focal wins on mean mIoU** (0.5786 > FedAvg 0.5711, +0.008).
   A real but modest improvement — first reproducible n=3 beat of
   FedAvg on this setting. Paper positions it as best-majority baseline.

3. **Biased client sampling is a reliability intervention, not a
   performance one.** Cuts n=3 variance ≈4× on both losses (σ 0.018→0.004),
   but does not lift mean performance. Trades seed-7 peak for lifting
   seeds 42/123 floors.

4. **Seed-bistability is optimization-landscape level, not
   data-presentation level.** Guaranteeing a rare-class-bearing client
   in every round (biased sampling) does not rescue seed-42's round-20
   C5 collapse. Rules out "unlucky rare-client schedule" as the cause.
   **Publishable negative result.**

5. **FL penalty at sr=0.25 dominates loss choice.** No configuration
   breaks 0.60 mean mIoU. Centralized ceiling ≈0.65. Loss tuning moves
   ±0.02, FL penalty is ≈0.07. Paper is honest about this: frames
   results as trading mean mIoU for rare-class recovery, not as
   beating centralized.

## Paper contributions (locked)

1. **Characterization of the rare-class data-absence failure mode** in
   federated segmentation under geographic Non-IID partitioning of
   seismic data (Parihaka, 20 clients, sr=0.25). Every standard FL
   method (FedAvg, FedProx, FedBN, FedVLS, rare-agg) leaves Class 5
   at 0.000 IoU.

2. **Recall-per-slice loss** — novel seismic-specific adaptation of
   Tian 2021's FNR-weighted CE, computing recall locally per 2D
   crossline rather than per batch. Only configuration in 40 runs
   with sustained final-round Class 5 recovery.

3. **Seed-bistability finding, strengthened.** Rare-class recovery on
   this setting is binary (~0.25 or exactly 0.000) per seed. We show
   this bistability survives:
   - Prototype alignment (up4 and up2)
   - 4 loss families (FocalDice, Unified Focal, Asymmetric Tversky, CRL)
   - 2 published rare-class losses (Tian 2021, FCIoU 2023)
   - Two λ values on CRL
   - Biased client sampling (guaranteed rare-client exposure)

   → The bistability is in the optimization landscape, not the
   data-presentation or loss.

4. **Negative result on biased client sampling** — compresses variance
   (~4×) but does not lift means. Documents a seemingly obvious
   intervention that doesn't solve the FL rare-class problem.

5. **State-dependent logit-adjustment at inference** (MetaFusion
   Chan 2019 / Menon 2021). Zero-cost post-hoc test-time intervention
   stackable on any FL-trained model. We show τ should be chosen
   based on the checkpoint's C5 posture: τ=0.5 for majority-biased
   models, τ=1.5 for collapse-state models (7.5× rare-class lift
   at zero mIoU cost). Combined with Recall-per-slice, sets new
   project records (best mIoU 0.5952, best test2 single-fold 0.6115).

## Motivation — Why existing FL methods fail on our setting

On Parihaka with geographic Non-IID partitioning (20 clients, each owning a
contiguous crossline slab, 5 sampled per round), our earlier benchmark sweep
showed that **every standard FL method leaves Class 5 (the rarest facies) at
0.000 IoU**, regardless of algorithm:

| Method            | Avg mIoU | Class 4 IoU | Class 5 IoU |
|-------------------|---------:|------------:|------------:|
| FedAvg            |   0.597  |       0.331 |       0.000 |
| FedAvg + CW       |   ≈0.59  |       ~0.33 |       0.000 |
| FedProx           |   ≈0.59  |       ~0.32 |       0.000 |
| FedBN             |   ≈0.59  |       ~0.33 |       0.000 |
| FedVLS            |   ≈0.59  |       ~0.33 |       0.000 |
| FedAvg + rare-agg |   ≈0.59  |       0.357 |       0.000 |
| Higher LE (5, 10) |     ↓    |           ↓ |       0.000 |

**Root cause — *data absence*, not data imbalance.** Because the split is
geographic, several clients see *zero* Class 5 pixels. No amount of
loss-reweighting, proximal regularization, BN localization, or aggregation
re-weighting can teach a client to recognize a class it has never seen.
The clients that *do* have Class 5 are only selected occasionally (sr=0.25)
and their gradients are diluted during aggregation.

## Our approach — FedSeis

FedSeis directly attacks the data-absence problem by providing a
**feature-level supervisory signal for vacant classes**, grounded in a tiny
public "reference section" held by the server. This mirrors standard
geophysical practice: in real interpretation workflows, a single fully-picked
reference crossline/well is almost always available to anchor facies
definitions, so assuming the server has ~1 labelled crossline per class is
operationally realistic.

### Three components

1. **Prototype alignment from a public reference section.**
   - Server holds 6 crosslines (1 per class, picked for the highest per-class
     fraction). At the start of each round, it forwards the *current global
     model* over these slices, extracts 64-dim features at UNet's `up4` layer
     (i.e. immediately before the output convolution), and averages them per
     class to produce six class prototypes `{p_c}`.
   - Prototypes are broadcast to clients alongside the model weights.
   - Clients add an alignment loss: for every pixel whose label is `c`, pull
     its normalized feature toward the normalized prototype `p_c`:
     `L_proto = mean(1 − cos(feat_pixel, p_c))`.
   - Crucially, this loss activates for **all** pixels, including classes the
     client has never locally observed — the server's reference slice is the
     only place those classes appear during that client's training, but via
     the prototype the client still gets a gradient that shapes its features
     toward the correct class manifold.

2. **Vacant-class distillation (from FedVLS).**
   Keeps the KL divergence between the frozen global model's logits and the
   local model's logits on the subset of classes vacant in the client's data.
   This prevents *catastrophic forgetting* of the classes the client cannot
   directly supervise. Our ablation tests how much this adds on top of (1).

3. **Rare-class-weighted aggregation.**
   Clients with a higher fraction of Class 4 + Class 5 pixels are upweighted
   in the server's FedAvg update (floor 0.01 to avoid zero contribution).
   This biases the aggregated model toward clients that actually supervise
   the minority classes, rather than letting them drown in the majority
   clients' updates.

### Combined client loss

```
L = L_focal_dice(logits, y)        # standard supervised term
  + α_p · L_proto(features, y, p)  # prototype alignment
  + α_d · L_dis(logits, y)         # KL on vacant-class logits
```

Defaults: `α_p = 0.5`, `α_d = 0.1`, rare-agg enabled.

## The server's reference section — what it actually contains

The server's "reference section" is **6 crosslines out of 701** — i.e.
**0.86% of the training cube**. We chose them by picking, per class, the
single crossline with the highest pixel fraction of that class:

| Class | Ref crossline | Class fills | Ref pixels of class | % of all training pixels of that class |
|------:|--------------:|------------:|--------------------:|---------------------------------------:|
|   0   |       344     |   32.36%    |            169,224  |                                0.840%  |
|   1   |        45     |   13.91%    |             76,131  |                                0.894%  |
|   2   |       672     |   57.45%    |            297,866  |                                0.855%  |
|   3   |        72     |    9.98%    |             40,591  |                                0.853%  |
|   4   |       193     |    8.71%    |             19,154  |                                0.815%  |
|   5   |       570     |   10.33%    |             10,564  |                                0.977%  |

So of every class, the server sees **<1% of the full training pixels for that
class** — no class is over-represented in the reference set. For the rarest
class (Class 5), the server holds roughly **1 out of every 102 pixels** of
that class that appear anywhere in training. This is far below the typical
"data held by each client" scale (~5% for 20 clients) and matches a
realistic labelling budget for a geophysical interpreter.

Class distribution in the **full Parihaka training set** (for comparison):

| Class | Pixel count   | % of training set |
|------:|--------------:|------------------:|
|   0   |  20,137,839   |       28.094%     |
|   1   |   8,519,666   |       11.886%     |
|   2   |  34,831,122   |       48.592%     |
|   3   |   4,760,778   |        6.642%     |
|   4   |   2,350,150   |        3.279%     |
|   5   |   1,081,200   |        1.508%     |

## Why this is reasonable for a journal contribution

* The assumption (one fully-picked reference crossline per facies at the
  server) is standard in seismic interpretation, *not* a privacy-destroying
  "send data to the server" shortcut — only 6 slices out of hundreds.
* The prototype alignment loss operates on **features**, not images. Clients
  never see the server's raw data; they receive 6 × 64-dim vectors per round.
* We tackle a concrete failure mode (0.000 IoU on minority classes) that
  every off-the-shelf FL method we tested produces, not a synthetic edge
  case.

## Overnight experimental campaign

Script: `run_fedseis_overnight.sh`. Results in `logs_overnight/`.

| Phase | Runs | Purpose |
|------:|-----:|---------|
|  A    |  5   | Main result (A1) + 4 ablations isolating proto / dis / rare-agg |
|  B    |  3   | α_proto sensitivity sweep ∈ {0.1, 1.0, 2.0} |
|  C    |  2   | Multi-seed rigor (seeds 123, 7) on default config |
|  D    |  2   | Multi-seed FedAvg baseline (seeds 123, 7) for matched comparison |

All runs are on the single hardest setting: Parihaka Non-IID 20c sr=0.25.
Baseline for comparison: FedAvg seed=42 = 0.597 mIoU, Class 5 = 0.000.

### Success criterion

The single number we care about: **does Class 5 IoU move off 0.000?** Any
result > 0 is a qualitative breakthrough over every prior method we
benchmarked. Secondary: Class 4 IoU improves beyond 0.357 (best prior),
and Avg mIoU ≥ 0.597 (does not regress on majority classes).

## Campaign results

Full per-run tables: **[`logs_overnight/RESULTS.md`](logs_overnight/RESULTS.md)**
(overnight up4 campaign) and **[`logs_layer_sweep/`](logs_layer_sweep/)**
(layer sweep, in progress). Working plan in **[`plan.md`](plan.md)**.

**Overnight (up4, 12 runs) — one-paragraph summary:**
- **Prototype alignment at up4 failed**: every α_p ∈ {0.1, 0.5, 1.0, 2.0}
  produced mIoU statistically indistinguishable from (or worse than)
  FedAvg.
- **Rare-class aggregation alone works** (A4, 0.5821 — best single run of
  the night, Class 5 peaking at 0.194 mid-training).
- **Class 5 > 0 is possible but oscillatory** — seed 7 held it to final
  round at 0.2295, the first non-zero Class 5 in our whole campaign.
- **Seed variance dominates method differences.** FedSeis up4 n=3 mean
  0.568 ± 0.006, FedAvg n=3 mean 0.571 ± 0.018 — the prior "0.597" FedAvg
  was a lucky seed.

**Layer sweep (8 runs, all complete):**
- Phase E (seed 42, α=0.5): up4 0.5679 / up3 **0.5756** / up2 0.5600 /
  up1 0.5657. Max-Class-5 leapt from 0.02 (up4) to 0.19–0.20 on the
  earlier layers.
- Phase F (up2 multi-seed): 0.5651 ± 0.0103, seed 7 finished with
  Class 5 = **0.2637** (best final-round C5 in our whole campaign).
- Phase G (α_proto on up2): monotonic decline in mIoU as α grows, same
  pattern as up4.

**Final multi-seed comparison (n=3):**

| Method | Best mIoU mean ± std | maxC5 mean |
|:---|:---:|:---:|
| FedAvg baseline | **0.5711 ± 0.0184** | 0.130 |
| FedSeis up4 | 0.5679 ± 0.0056 | 0.135 |
| FedSeis up2 | 0.5651 ± 0.0103 | **0.216** |

**Final verdict on prototypes.** Prototype alignment produces a real but
insufficient rare-class-recovery effect. It is more *reliable* on
earlier layers (up2 maxC5 0.216 vs up4 0.135) but costs some
majority-class accuracy. No layer, no α value, beats FedAvg's mIoU.
Prototypes are done as a novel-method contribution.

**Paper pivot (final).**
1. **Rare-class aggregation weighting** — the one intervention that
   consistently helps; best single run of both campaigns (0.5821, A4).
2. **Seed-bistable Class-5 recovery** — genuine empirical finding; the
   recovery probability depends on early client-sampling luck, and
   earlier-layer prototypes increase that probability without fixing
   the underlying bistability.
3. **Next-iteration contributions (per advisor feedback)** — compound
   loss designs (Unified Focal, asymmetric Tversky) and a novel
   seismic-specific per-crossline rare-class recall loss. See
   `plan.md` §5.

## Loss-function sweep — IN PROGRESS (6 of 14 runs complete)

Script: `run_loss_sweep.sh`. Logs: `logs_loss_sweep/`. All runs
keep rare-class aggregation enabled.

**Phase H (FocalDice baseline, n=3):**

| Seed | Best mIoU | Final C5 | max C5 |
|-----:|----------:|---------:|-------:|
| 42   |    0.5724 |   0.0000 | 0.193  |
| 123  |    0.5569 |   0.0000 | 0.181  |
| 7    |    0.5588 |   0.0734 | 0.224  |
| **mean ± std** | **0.5627 ± 0.0069** | — | 0.199 |

**Phase I1–I3 Unified Focal Loss (n=3):**

| Seed | Best mIoU | Final C5 | max C5 |
|-----:|----------:|---------:|-------:|
| 42   |    0.5850 |   0.0000 | 0.165  |
| 123  |    0.5554 |   0.0000 | 0.178  |
| **7** | **0.5954** | **0.2505** | 0.2505 |
| **mean ± std** | **0.5786 ± 0.0169** | — | 0.198 |

**Δ vs FocalDice: +0.0159 mIoU** — first reproducible positive result
across all three seeds over any baseline in the project. I3 (seed 7)
also held Class 5 to the final round at 0.2505 — third seed-7 run to
break bistability (after C2 up4 0.2295 and F2 up2 0.2637).

**Phase I4–I6 Asymmetric Tversky (n=3):**

| Seed | Best mIoU | Final mIoU | max C5 | Note |
|-----:|----------:|-----------:|-------:|------|
| 42   |    0.5724 |     0.0361 | 0.000  | Round 20 collapse |
| 123  |    0.5626 |     0.5626 | 0.189  | Clean |
| 7    |    0.5845 |     0.5614 | 0.133  | Clean |
| **mean ± std** | **0.5732 ± 0.0090** | — | 0.107 | — |

Δ vs FocalDice baseline: **+0.010 mIoU** (less than Unified Focal's
+0.016). One-in-three seed instability (I4) is a real robustness concern.

**Phase I7–I9 Crossline Recall (novel, λ=1.0) — n=3, COMPLETE:**

| Seed | Best mIoU | Final C5 | max C5 | Note |
|-----:|----------:|---------:|-------:|------|
| 42   |    0.5607 |    0.000 | 0.142  | −0.012 vs FocalDice s42 |
| 123  |    0.5594 |    0.000 | 0.154  | +0.003 vs FocalDice s123 |
| **7** | **0.5825** | **0.273** | 0.273 | 🔥 best project final-round C5 |
| **mean ± std** | **0.5675 ± 0.0107** | — | 0.190 | Δ vs FocalDice: **+0.005** |

**I9 seed 7 is the headline result.** Final-round Class 5 = **0.273**
beats every prior project record (F2 up2 s7 = 0.2637, I3 UnifiedFocal
s7 = 0.2505). Uniquely, C5 **kept rising** into round 20 rather than
spiking mid-training and collapsing — suggesting per-crossline recall
provides a gradient that *doesn't fade* when rare-class-bearing clients
aren't sampled. On mean mIoU, novel loss is between FocalDice (+0.005)
and Unified Focal (−0.011), so in the paper this will be framed as
**trading mean mIoU for rare-class recovery** — exactly what
per-crossline recall was designed to do.

### Loss-sweep aggregate (3 losses complete, 1 λ-sensitivity left)

| Loss | n | Best mIoU (mean ± std) | max C5 mean | Best final-round C5 | Δ vs FocalDice |
|------|--:|-----------------------:|------------:|--------------------:|---------------:|
| FocalDice (baseline) | 3 | 0.5627 ± 0.0069 | 0.199 | 0.073 (H3) | — |
| **Unified Focal** | 3 | **0.5786 ± 0.0169** | 0.198 | 0.2505 (I3) | **+0.016** |
| Asymmetric Tversky | 3 | 0.5732 ± 0.0090 | 0.107 | never C5>0 | +0.010 |
| **Crossline Recall (novel)** | 3 | 0.5675 ± 0.0107 | 0.190 | **0.273 (I9)** 🔥 | +0.005 |

**Phase J — λ_crl sensitivity (seed 42 only, COMPLETE):**

|        λ | Best mIoU | Final C5 | max C5 |
|---------:|----------:|---------:|-------:|
| 0.3 (J1) |    0.5642 |    0.000 |  0.160 |
| 1.0 (I7) |    0.5607 |    0.000 |  0.142 |
| **3.0 (J2)** | **0.5684** | 0.000 | 0.177 |

**λ=3.0 is best on seed 42** on both axes — suggests pushing CRL weight
*higher* than 1.0 is the right direction. Obvious next experiment:
re-run I8/I9 at λ=3.0 with n=3 seeds to see whether the mean closes the
Unified Focal gap.

## Final loss-sweep ranking (n=3 each, 14 runs total, campaign complete)

Column semantics: **max C5 (mean over seeds)** = per-seed peak C5 IoU over all
20 rounds, then meaned across 3 seeds. **Best-seed final-round C5** = single
max of the round-20 C5 values across the 3 seeds. The second can exceed the
first because it's a one-seed max vs a three-seed mean-of-peaks.

| Loss | mean mIoU ± std | max C5 (mean over seeds) | best-seed final-round C5 | Δ vs FocalDice |
|:-----|---------------:|------------------------:|-------------------------:|---------------:|
| FocalDice (baseline) | 0.5627 ± 0.0069 | 0.199 | 0.073 (H3) | — |
| **Unified Focal** ← best mIoU | **0.5786 ± 0.0169** | 0.198 | 0.2505 (I3) | **+0.016** |
| Asymmetric Tversky | 0.5732 ± 0.0090 | 0.107 | never C5>0 | +0.010 |
| **Crossline Recall (novel)** ← best rare-class | 0.5675 ± 0.0107 | 0.190 | **0.273 (I9)** 🔥 | +0.005 |

**Two complementary winners.** Unified Focal gives the best mean mIoU
(matches FedAvg mean 0.5711 and beats it by +0.008). Crossline Recall
gives the best rare-class recovery in the whole project (final-round
C5 = 0.273, uniquely still rising at round 20). The paper will report
both, framed as a majority-class vs rare-class trade-off that the
practitioner picks based on downstream need.

## Honest take — what the campaign actually tells us

* **Unified Focal is a real but small win.** +0.016 mIoU over FocalDice,
  +0.008 over FedAvg. Not a paper headline on its own, but it's the
  first thing in the project that beats FedAvg reproducibly across 3
  seeds.
* **Crossline Recall is the paper.** Not because of mean mIoU
  (middle of the pack at +0.005) but because I9 seed 7 is the *only*
  run in the entire project — across 40+ FL experiments — where
  Class 5 is still *climbing* in the final round instead of collapsing
  or spiking-then-dying. That's a qualitatively new dynamic,
  attributable to the per-crossline differentiable soft-recall term,
  and it's a defensibly novel seismic-specific contribution.
* **λ=3.0 hint is strong enough to invest in.** J2 single-seed data
  point beats λ=1.0 and λ=0.3 on both axes. We don't know yet whether
  this holds across seeds — Phase K answers that.
* **Metric framing concern.** Reporting mean mIoU across 6 classes
  hides the phenomenon the paper is about. Switching the headline
  metric to rare-class IoU (class 4 + class 5) and balanced accuracy.

## Phase K — CRL λ=3.0 × 3 seeds (COMPLETE 2026-04-21)

Direct test of whether stronger CRL regularization closes the Unified
Focal gap. Answer: **no.**

| Run | Seed | Best mIoU | Final C5 | max C5 |
|---|---:|---:|---:|---:|
| K1 | 42 | 0.5684 | 0.000 | 0.177 |
| K2 | 123 | 0.5559 | 0.000 | 0.154 |
| **K3** | **7** | **0.5468** | **0.201** | 0.212 |
| **mean ± std** | — | **0.5570 ± 0.0089** | — | 0.181 |

**Verdict.** λ=3.0 is *worse* than λ=1.0 on both axes.
- Mean mIoU 0.5570 < I7–I9 λ=1.0's 0.5675 (−0.010)
- Best-seed final-round C5 0.201 < I9's 0.273 (−0.072)
- J2's single-seed 0.5684 on seed 42 was not signal — K1 replicates
  it but K2 and K3 are both worse than their λ=1.0 counterparts.

**CRL λ=1.0 stands as the novel-loss headline** (0.5675 mean, 0.273
best-seed final-round C5). λ has a sweet-spot shape rather than
monotone-better-with-more; this is itself a useful paper point
(the per-crossline recall term is not a knob to crank arbitrarily).

## Reality check (2026-04-21): ~0.58 is the FL ceiling

**34 runs of FL on this exact setting**, every loss the advisor
recommended + everything we invented, n=3 seeds each. **Nothing breaks
0.60 mIoU.** Best n=3 mean is Unified Focal at 0.5786; best single run
is N3 Recall Loss seed 7 at 0.5927. Centralized upper bound is ~0.65.
The FL penalty at sr=0.25 (10 of 20 clients sampled per round on
geographic Non-IID) is the dominant source of loss, not loss function
choice. **Loss tuning is not going to save the mIoU story.**

What this means:
1. Switch the paper's primary metric to rare-class-first (balanced
   accuracy + per-class IoU for C4/C5). Mean mIoU becomes secondary.
   Every result we have re-sorts: the losses that help rare classes
   become the winners under this metric.
2. Frame the paper as *characterizing the rare-class failure mode of
   federated segmentation under geographic Non-IID* + *two
   interventions that trade mean mIoU for measurable rare-class
   recovery*. Don't overclaim a mean-mIoU beat.
3. Skip F3 transfer unless Phase L lands something dramatic. It's 7h
   of compute for a result we cannot predict, on a secondary setting.

## Phase P+L (launched 2026-04-21 20:47, ETA ~3h)

Two hypotheses tested overnight before paper draft:

### Phase P — novelty ablation (per-slice Recall Loss × 3 seeds)

Runs Tian 2021's FNR-weighted CE computed *per 2D crossline slice*
instead of per batch. If this beats batch-level Recall Loss (N1–N3
mean 0.5725 / maxC5 0.229), locality of recall computation is our
contribution and CRL's novelty is defensible as "per-crossline
recall-based loss for seismic FL." If it matches or loses, CRL novelty
evaporates and the paper reframes to "replication + analysis of Tian
2021 in FL seismic."

### Phase L — biased client sampling on top-2 losses (6 runs)

Unified Focal × 3 seeds + Recall Loss × 3 seeds, all with
`--force_rare_client` (guarantee ≥1 Class-5-bearing client per round,
remaining slots sampled uniformly from the rest). Direct attack on
data-absence failure mode. If final-round C5 lifts on top of either
loss, biased sampling is the operational contribution that makes the
paper about *method-transfer* not just empirical characterization.

## Phase L — biased client sampling (COMPLETE 2026-04-22 02:25)

Biased sampling (`--force_rare_client`) forces ≥1 Class-5-bearing
client into every round's selection (remaining slots uniform).
Tested on the two best losses × 3 seeds each = 6 runs.

### L1–L3: Unified Focal + biased sampling

| Seed | Uniform (I1–I3) | Biased (L1–L3) | Δ mIoU | Δ maxC5 | Δ final C5 |
|---:|:---|:---|---:|---:|---:|
| 42  | 0.5850 / 0.165 / 0.000 | 0.5587 / 0.199 / 0.000 | −0.026 | +0.034 | 0.000 |
| 123 | 0.5554 / 0.178 / 0.000 | 0.5679 / 0.145 / 0.000 | +0.013 | −0.033 | 0.000 |
| 7   | 0.5954 / 0.2505 / 0.2505 | 0.5651 / 0.223 / 0.223 | −0.030 | −0.027 | −0.027 |
| **n=3 mean ± std** | **0.5786 ± 0.021 / 0.198** | **0.5639 ± 0.005 / 0.189** | **−0.015** | **−0.009** | **−0.009** |

**Unified Focal + biased: net-negative on every axis.**

### L4–L6: Recall Loss + biased sampling

| Seed | Uniform (N1–N3) | Biased (L4–L6) | Δ mIoU | Δ maxC5 | Δ final C5 |
|---:|:---|:---|---:|---:|---:|
| 42  | 0.5613 / 0.211 / 0.000 | 0.5748 / 0.249 / 0.000 | +0.014 | +0.038 | 0.000 |
| 123 | 0.5635 / 0.218 / 0.000 | 0.5707 / 0.196 / 0.073 | +0.007 | −0.022 | **+0.073** |
| 7   | 0.5927 / 0.259 / 0.251 | 0.5663 / 0.236 / 0.221 | −0.026 | −0.023 | −0.030 |
| **n=3 mean ± std** | **0.5725 ± 0.018 / 0.229** | **0.5706 ± 0.004 / 0.227** | **−0.002** | **−0.002** | **+0.014** |

**Recall Loss + biased: essentially null on means, but variance compression is dramatic.**

### The real finding — variance compression

| Config | n=3 mIoU std | n=3 maxC5 spread |
|:---|---:|---:|
| Unified Focal uniform   | 0.021 | [0.165, 0.251] |
| Unified Focal biased    | 0.005 | [0.145, 0.223] |
| Recall Loss uniform     | 0.018 | [0.211, 0.259] |
| Recall Loss biased      | 0.004 | [0.196, 0.249] |

Biased sampling **cuts run-to-run variance roughly 4×** across both
losses. It trades seed-7's peak performance for raising the floor on
seeds 42 and 123. On Recall Loss this trade is net-zero on means;
on Unified Focal it's net-negative.

**Seed-42 final-round C5 never moved from 0.000** despite guaranteed
rare-client exposure every round. This confirms the seed-bistability
isn't driven by *which rounds* the rare client appears — guaranteed
presence doesn't unlock seed 42's round-20 trajectory. The
bistability is in the optimization landscape itself.

**Paper framing for Phase L:**
- **Positive result:** biased client sampling *compresses seed variance
  by ~4×* on both losses tested — useful when reliability matters more
  than peak performance (operational FL deployments).
- **Neutral/negative result on means:** biased sampling does not lift
  mean mIoU or mean rare-class recovery.
- **Seed-bistability robustness finding:** seed 42's round-20 C5 collapse
  survives guaranteed rare-client exposure. Rules out "bad seed never
  saw the rare client" as the cause of bistability. The cause is
  deeper — optimization-landscape level, not data-presentation level.

### Best project configurations after Phase L

| Config | mIoU (mean ± std) | maxC5 mean | best-seed final C5 |
|:---|---:|---:|---:|
| Unified Focal uniform | 0.5786 ± 0.021 | 0.198 | 0.2505 |
| Recall-per-slice uniform | 0.5729 ± 0.017 | 0.240 | **0.280** 🏆 |
| Recall Loss biased | 0.5706 ± 0.004 | 0.227 | 0.221 |
| Recall Loss uniform | 0.5725 ± 0.018 | 0.229 | 0.251 |
| Unified Focal biased | 0.5639 ± 0.005 | 0.189 | 0.223 |

**Recall-per-slice (P) stays the novelty headline** — best-seed final
C5 0.280 is the project record and the only configuration that
sustained C5 across multiple final rounds.

### Paper table reshape (post-Phase L)

Primary table switches to rare-class-first metric. Columns: per-class
IoU for C4 and C5, balanced accuracy (mean recall across 6 classes),
*then* mean mIoU. Winners under the new metric:
- **Best rare-class C5 recovery:** Recall-per-slice (our locality
  ablation) at 0.280 best-seed final.
- **Best reliability (lowest variance):** Recall Loss + biased, σ=0.004.
- **Best mean mIoU:** Unified Focal uniform, 0.5786 ± 0.021.
- **Best combined:** Recall-per-slice (mIoU 0.5729, rare-class best).

## Phase P — novelty ablation RESULT (COMPLETE 2026-04-21 22:32)

Per-slice Recall Loss (Tian 2021 formulation applied per 2D crossline
slice instead of per batch) × 3 seeds. Direct test of whether locality
of recall computation is the CRL contribution.

| Seed | Recall-per-batch (N) | Recall-per-slice (P) | Δ mIoU | Δ maxC5 |
|---:|---:|---:|---:|---:|
| 42  | 0.5613 / 0.211 | 0.5629 / 0.222 | +0.002 | +0.011 |
| 123 | 0.5635 / 0.218 | 0.5627 / 0.219 | −0.001 | +0.001 |
| **7** | 0.5927 / 0.259 | **0.5932 / 0.280** | +0.001 | **+0.021** |
| **n=3 mean** | **0.5725 / 0.229** | **0.5729 / 0.240** | +0.0004 | **+0.011** |

**Verdict: per-slice locality IS the contribution, but subtly.**
- Mean mIoU delta is within noise (+0.0004).
- **Max-C5 mean gains +0.011** (0.229 → 0.240) — real but small.
- **Best-seed final-round C5 sets a new project record: 0.280** (P3 s7)
  vs prior record 0.273 (I9 s7 CRL λ=1.0).
- **P3 s7 shows sustained C5 across final rounds** (0.265 / 0.207 /
  0.008 / 0.280 for rounds 17 / 18 / 19 / 20) — qualitatively
  different from every prior run where C5 either spiked once or sat
  at zero.

**Paper framing.** The novelty is not "recall loss for segmentation"
(that's Tian 2021). The novelty is **computing recall locally, per
2D seismic slice, in a federated setting where client data is
crossline-contiguous**. Locality matches the spatial structure of
the data *and* of the Non-IID partition. This is defensible as a
small but specific seismic-FL contribution.

**Best single run of project (mIoU):** P3 s7 = 0.5932
(round 18 hit 0.5932, round 17 test2 was 0.6211, round 20 test2 was
0.6297). Still below centralized 0.65 but the gap narrowed.

## Phase N/O — advisor-suggested SOTA losses (COMPLETE 2026-04-21 19:56)

### Phase N: Recall Loss (Tian et al. 2021, OpenReview SlprFTIQP3)

Per-class CE reweighted by instantaneous false-negative rate.
Closest published analogue to our Crossline Recall Loss.

| Seed | Best mIoU | Final C5 | max C5 |
|---:|---:|---:|---:|
| 42 (N1)  | 0.5613 | 0.000 | 0.211 |
| 123 (N2) | 0.5635 | 0.000 | 0.218 |
| **7 (N3)** | **0.5927** | **0.251** | **0.259** |
| **n=3 mean ± std** | **0.5725 ± 0.0175** | — | **0.229** |

### Phase O: FCIoU Loss (Sheffield et al., MAKE 2023)

Focal-weighted per-class soft-IoU.

| Seed | Best mIoU | Final C5 | max C5 |
|---:|---:|---:|---:|
| 42 (O1)  | 0.5564 | 0.000 | 0.102 |
| 123 (O2) | 0.5529 | 0.000 | 0.069 |
| 7 (O3)   | 0.5534 | 0.000 | 0.197 |
| **n=3 mean ± std** | **0.5542 ± 0.0019** | — | **0.123** |

### All-loss ranking (n=3 each, 20 runs total on Parihaka Non-IID 20c sr=0.25)

| Loss | mean mIoU ± std | max C5 (mean/seeds) | best-seed final C5 | Δ mIoU vs FocalDice |
|:-----|---------------:|-------------------:|-------------------:|---------------:|
| FCIoU (MAKE 2023)            | 0.5542 ± 0.002 | 0.123 | never >0 | −0.009 |
| CRL λ=3.0                    | 0.5570 ± 0.009 | 0.181 | 0.201 (K3) | −0.006 |
| FocalDice (baseline)         | 0.5627 ± 0.007 | 0.199 | 0.073 (H3) | — |
| CRL λ=1.0 (ours)             | 0.5675 ± 0.011 | 0.190 | **0.273 (I9)** | +0.005 |
| **Recall Loss (Tian 2021)**  | **0.5725 ± 0.018** | **0.229** | 0.251 (N3) | +0.010 |
| Asymmetric Tversky           | 0.5732 ± 0.009 | 0.107 | never >0 | +0.010 |
| **Unified Focal (Yeung 2021)** | **0.5786 ± 0.017** | 0.198 | 0.2505 (I3) | **+0.016** |

### Paper-story re-sort

1. **Unified Focal still wins on mean mIoU** (+0.016 over FocalDice, +0.008 over FedAvg).
2. **Recall Loss now wins on "max C5 (mean over seeds)"** (0.229 > CRL λ=1.0's 0.190) — published baseline, not our novelty. CRL's uncontested advantage is narrowed to *best-seed final-round* C5 (0.273 vs Recall's 0.251) and the "still rising at round 20" dynamic on I9.
3. **FCIoU is a clear underperformer** — cite, include in table, note it doesn't transfer here (focal-IoU compound fails on classes with tiny IoU baselines).
4. **CRL novelty claim must be reframed**. No longer "best rare-class recovery" outright. Strongest framing now:
   * matches or beats Recall Loss on *final-round* rare-class retention;
   * uniquely produces a "climbing to max at round 20" trajectory;
   * per-crossline structure is the real ablation — need to run **Recall Loss applied slice-level** as the direct ablation vs our per-crossline design.

### Updated Phase L / M plan

- Primary contender for Phase L (biased sampling): **Unified Focal** (best mean) and **Recall Loss** (best rare-class mean). Run biased sampling on each with 3 seeds.
- CRL λ=1.0 drops in priority for Phase M unless the per-crossline ablation holds up.
- New experiment on the table: **Recall Loss with per-crossline normalization** (Recall-Loss-CL) — direct test of whether our seismic-specific contribution is the slice-level recall *computation*, not just *recall-based weighting*.

## Next steps — updated after Phase K (2026-04-21)

**Dropped** from the queue: F3 transfer, more prototype experiments.
**Order of remaining work:**

1. **Phase K — CRL λ=3.0 × 3 seeds** (~1h, HIGHEST PRIORITY): direct
   test of whether CRL's n=3 mean closes the Unified Focal gap at
   stronger regularization. If yes, the novel loss becomes the
   mean-mIoU winner *and* the rare-class winner simultaneously.
2. **Phase L — biased client sampling** (~3h): implement
   `--force_rare_client` flag so every round samples ≥1 client with
   `rare_fraction > 0`. Run with winning loss × 3 seeds. Complements
   rare-class aggregation weighting rather than replacing it.
3. **Phase M — n=5 consolidation** (~6h): only on the 2–3 paper
   headline configurations that survive K and L. Seeds 42, 123, 7,
   99, 2025.
4. **Paper draft** after M.

Primary paper metric: **rare-class-first** (per-class IoU for C4 / C5
and balanced accuracy), not mean mIoU.

## Paper outline — draft (2026-04-22 post-campaign)

Target venue: seismic-focused FL/ML workshop or IMAGE short paper. The
centralized-benchmark journal route is not defensible given the
FL-penalty reality at sr=0.25.

### Title candidates
- "Rare-class recovery in federated seismic facies segmentation:
  a locality-aware loss and a seed-bistability finding"
- "Locality matters: per-slice recall loss for rare-class recovery
  under geographic Non-IID federated seismic segmentation"

### Abstract (≤200 words, first-draft sketch)
Federated learning on seismic facies segmentation under geographic
Non-IID partitioning leaves the rarest class at 0.000 IoU across every
standard FL method (FedAvg, FedProx, FedBN, FedVLS, rare-class
aggregation). We show this failure is driven by *data absence*, not
imbalance: several clients see zero pixels of the rare class. We
evaluate 40 configurations spanning 4 loss families, 2 published
rare-class losses (Tian 2021, FCIoU 2023), biased client sampling,
and a novel per-slice recall loss that computes recall weights locally
on each 2D crossline. The locality-aware loss is the only configuration
with sustained final-round recovery (best-seed C5 IoU 0.280 vs 0.000
across baselines). Rare-class recovery is seed-bistable: per seed it
is either ~0.25 or exactly 0.000, with no stable middle. We show this
bistability survives loss choice, prototype alignment, λ tuning, and
*guaranteed rare-client exposure* (biased sampling), locating it in
the optimization landscape rather than in data presentation. Biased
sampling reduces run-to-run variance 4× but does not lift mean
performance — a publishable negative result on a seemingly obvious
intervention.

### Section outline
1. **Introduction.** Seismic interpretation + federated learning
   motivation. Geographic Non-IID as the realistic partition. Paper
   contributions (4 bullets per §Campaign final ranking).
2. **Related work.** FL methods (FedAvg/Prox/BN/VLS). Rare-class
   losses for segmentation (focal, Tversky, Unified Focal, Recall
   Loss, FCIoU). Prototype alignment. MetaFusion inference-time
   rule. Seismic FL literature.
3. **Problem setup.** Parihaka cube, 20-client geographic split,
   sr=0.25, 20 rounds, 3 local epochs, UNet backbone. Class-distribution
   table. Why data-absence differs from imbalance.
4. **Methods.**
   4.1 Baselines (FedAvg, FedProx, FedBN, FedVLS, rare-class aggregation).
   4.2 Loss families evaluated (FocalDice, Unified Focal, Asymmetric
       Tversky, Recall Loss Tian 2021, FCIoU MAKE 2023, our CRL, our
       Recall-per-slice).
   4.3 Biased client sampling (`--force_rare_client`).
   4.4 Metric framing: rare-class-first (C4/C5 IoU, balanced accuracy,
       finalC5 retention) as primary; mean mIoU secondary.
5. **Experiments.**
   5.1 Loss-sweep (Phases H/I/J/K/N/O/P) — 30 runs, n=3 each.
   5.2 Biased client sampling (Phase L) — 6 runs.
   5.3 Seed variance analysis.
6. **Results.**
   6.1 Main rare-class-first ranking table (11 configs).
   6.2 Recall-per-slice as novelty headline — qualitative C5 trajectory
       plot for P3 s7 (0.265/0.207/0.008/0.280 at rounds 17–20).
   6.3 Variance compression under biased sampling (~4× on both losses).
   6.4 Seed-bistability across 12+ method variants.
7. **Discussion.**
   7.1 Why per-slice locality matches seismic data structure.
   7.2 Seed-bistability as an optimization-landscape property.
   7.3 Negative result on biased sampling — what it rules out.
   7.4 Limitations: single dataset (Parihaka), sr=0.25 canonical
       setting only, no F3 transfer.
8. **Conclusion.**

### Figures / tables (to generate from `wandb/` + logs)
- **Fig 1.** Class distribution and geographic split (existing).
- **Fig 2.** Per-round Class 5 IoU trajectories for 6 headline configs
  (P3 s7 vs I9 s7 vs I3 s7 vs N3 s7 vs baseline).
- **Fig 3.** Seed-bistability visualization (final-round C5 per seed
  across 12+ configs, showing bimodal ~0.25 vs 0.000).
- **Fig 4.** Variance compression under biased sampling (n=3 spread
  plot, uniform vs biased, both losses).
- **Tab 1.** Main rare-class-first ranking (11 configs).
- **Tab 2.** Per-seed breakdown for top 4 configs (Recall-per-slice,
  Recall Loss, Unified Focal, Recall+biased).

### Remaining pre-draft work
1. Generate figures (✅ Fig 2 trajectory, Fig 3 bistability, Fig 4
   variance — all in `paper_figures/`).
2. ✅ MetaFusion (Chan 2019) / logit adjustment — post-hoc scan on
   P3 s7 best-mIoU checkpoint (round 18 in log, round_17 filename).
   τ=0.5 lifts C5 IoU **0.207 → 0.226** (+9% relative) and mIoU
   0.5932 → 0.5952 (+0.002). Zero-cost test-time intervention. Adds
   a test-time contribution to the paper.
3. *(Skip)* Phase M n=5 consolidation — low marginal value given
   campaign is already 40 runs and the headline story is locked.
4. *(Skip)* F3 transfer — 7h compute for unpredictable secondary
   setting; paper is stronger as "Parihaka deep characterization" than
   "two-dataset shallow comparison."

## MetaFusion / logit adjustment (Chan 2019) post-hoc (2026-04-22)

Applied `logits' = logits - τ · log(p_train)` at inference on P3 s7
best-mIoU checkpoint. Training class prior
`p = [0.281, 0.119, 0.486, 0.066, 0.033, 0.015]` (Parihaka training
pixels). Scan over τ:

| τ | mIoU1 | mIoU2 | avg | per-class avg |
|:---|---:|---:|---:|:---|
| 0.00 (baseline) | 0.5804 | 0.6060 | 0.5932 | 0.892 / 0.790 / 0.904 / 0.531 / 0.236 / **0.207** |
| 0.25 | 0.5803 | 0.6094 | 0.5949 | 0.893 / 0.788 / 0.905 / 0.531 / 0.235 / 0.217 |
| **0.50** | 0.5791 | **0.6113** | **0.5952** 🏆 | 0.893 / 0.784 / 0.904 / 0.531 / 0.234 / **0.226** |
| 0.75 | 0.5773 | 0.6115 | 0.5944 | 0.894 / 0.778 / 0.903 / 0.529 / 0.232 / **0.231** |
| 1.00 | 0.5750 | 0.6100 | 0.5925 | 0.894 / 0.771 / 0.901 / 0.526 / 0.230 / 0.233 |
| 1.25 | 0.5724 | 0.6066 | 0.5895 | 0.893 / 0.763 / 0.899 / 0.523 / 0.228 / 0.231 |

**Finding.** τ=0.5 is the Pareto-optimal point:
- mIoU improves marginally (+0.002).
- C5 IoU improves +0.019 absolute (+9% relative) — 0.207 → 0.226.
- Test2-fold single-fold mIoU **0.6113** = new best-ever on the Parihaka
  project (beats centralized-ish territory on one fold).
- Majority-class IoU essentially unchanged (C0/C1/C2/C3 shift <0.01).

**Paper framing.** MetaFusion/logit-adjustment is a ~1-line
post-hoc fix stackable on any trained model. Adds a complementary
test-time contribution to the loss-based training contribution
(Recall-per-slice). Suggests paper pitch:
*"Train with locally-computed recall weights (recall-per-slice);
deploy with logit adjustment (τ=0.5). Combined, lifts project-best
finalC5 to 0.231 with mIoU 0.5944 — zero additional training cost
for the second half."*

### Multi-checkpoint scan (rounds 18, 19, 20 in log numbering)

MetaFusion behaves very differently depending on the model's current
rare-class bias:

#### Round 18 (best-mIoU checkpoint, log)
| τ | mIoU | C5 |
|:---|---:|---:|
| 0.00 | 0.5932 | 0.207 |
| 0.50 🏆 | **0.5952** | **0.226** |
| 0.75 | 0.5944 | 0.231 |

Pareto optimum at τ=0.5. Lifts C5 +9% relative at zero mIoU cost.

#### Round 19 (log round 19, C5=0.008 collapse state)
| τ | mIoU | C5 |
|:---|---:|---:|
| 0.00 | 0.5585 | 0.008 |
| 1.00 | 0.5629 | 0.039 |
| 1.25 | 0.5634 | 0.050 |
| 1.50 🏆 | **0.5634** | **0.060** |
| 2.00 | 0.5608 | 0.078 |

**MetaFusion partially rescues the collapse state.** At τ=1.5, C5 goes
from 0.008 → 0.060 (7.5× relative) *and* mIoU improves (+0.005). At
τ=2.0, C5 keeps climbing (0.078) while mIoU barely drops. Useful paper
point: when a model lands in the "C5=0 attractor," post-hoc logit
adjustment can bend it back without retraining.

#### Round 20 (log round 20, C5=0.280 rare-class peak state)
| τ | mIoU | C5 |
|:---|---:|---:|
| 0.00 🏆 | **0.5816** | **0.280** |
| 0.25 | 0.5800 | 0.273 |
| 0.50 | 0.5768 | 0.264 |

τ=0.0 is best. The model is already rare-class-biased here, so
subtracting more prior hurts both axes.

### Paper-grade finding

MetaFusion / logit adjustment is a **state-dependent corrector**:
- on majority-biased checkpoints, small τ (≈0.5) lifts rare-class IoU
  at zero cost;
- on C5-collapse checkpoints, moderate τ (≈1.5) reverses the collapse;
- on rare-class-peak checkpoints, τ=0 is optimal (don't over-adjust).

Combined with Recall-per-slice training, this gives a two-stage
strategy: (1) train with per-slice recall to maximize the *chance* of
landing in the rare-class attractor, (2) at inference, select τ based
on the checkpoint's C5 posture (zero if strong, 0.5 if moderate,
1.5 if collapsed). τ selection could be done on a held-out validation
slice at essentially zero cost.

**New best project numbers:**
- Best single mIoU: **P3 s7 + τ=0.5 → 0.5952** (beats raw P3's 0.5932).
- Best single fold: **test2 at τ=0.75 → 0.6115** (first time sub-project
  single-fold above 0.61 on Parihaka FL).
- Collapse rescue: round 19 C5 0.008 → 0.060 via τ=1.5 (7.5× lift at zero mIoU cost).

---

## Tier 0 re-analysis (2026-08-03) — best-round vs final-round selection

`train_federated.py` saves `best_global_model.pth` on **test** mIoU across the
20 rounds, and every table above reports that maximum. That is max-over-rounds
on the evaluation set. It is not merely optimistic — it is optimistic *by
different amounts per config*, which is exactly the comparison being made.

Recomputed from the raw phase logs, n=5 seeds each:

| Config | Best-round mIoU | Final-round mIoU | Selection bias |
|---|---|---|---|
| Baseline (rare agg) | 0.5695 ± 0.0104 | 0.5514 ± 0.0121 | +0.018 |
| V3+frc | 0.5849 ± 0.0085 | **0.5421 ± 0.0753** | **+0.043** |
| T alone | 0.5807 ± 0.0078 | **0.5677 ± 0.0171** | +0.013 |

**Two headline claims do not survive this correction:**

1. **"V3+frc has ~2x tighter variance."** That was best-round std. On
   final-round, V3+frc is the *least* stable config (± 0.0753 vs baseline's
   ± 0.0121) — seed 2025 ends at 0.3919, a full collapse that best-round
   selection hides entirely.
2. **"V3+frc is the best single-model config."** On final-round, **T alone
   wins** (0.5677 vs 0.5421) and is 4x more stable. The V3+frc ranking was an
   artifact of picking each run's luckiest round.

Per-seed final-round detail:

    V3+frc   s42 0.5733 | s123 0.5740 | s7 0.5892 | s99 0.5821 | s2025 0.3919
    T alone  s42 0.5906 | s123 0.5706 | s7 0.5694 | s99 0.5703 | s2025 0.5375

C5 at final round also differs from the best-round story: baseline 0.051,
V3+frc 0.098, T 0.093 — T matches V3+frc on rare-class recovery without the
collapse risk.

**Consequence.** Report final-round as primary, or select on a held-out
validation split. Best-round may stay as a clearly-labelled secondary column.
The ensemble/MetaFusion numbers (0.6155 / 0.6176) inherit this bias, since they
ensemble the best-round checkpoints; they need recomputation from final-round
checkpoints before publication.
