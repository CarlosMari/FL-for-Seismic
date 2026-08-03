# IMAGE 2026 — Oral Presentation Planning

**Paper (accepted):** *The Effectiveness of Federated Learning in Seismic Interpretation*
**Format:** Oral, 20 min (15–17 talk + 3–5 Q&A). Submitted ~5 months ago, so
**new results may be presented.**
**Updated:** 2026-08-03, after the Tier 0/1 campaign and the final-round recompute.

> ## REPORTING CONVENTION — state this once, early, then never mix
>
> **All numbers below are FINAL-ROUND (round 20) / FINAL-EPOCH.** Both the
> federated and the centralized code select a "best" checkpoint by maximum
> **test** mIoU across rounds/epochs, which leaks the evaluation set and
> flatters unstable configurations most. Best-round figures appear only where
> explicitly labelled, as the *biased* comparison.
>
> Consequences: our ensemble headline is **0.5965**, not 0.6155; the centralized
> ceiling is **~0.62**, not 0.693. Correcting both leaves the story *stronger*,
> because the ceiling was more inflated than our own numbers.

---

## THE BIGGEST FINDING

> **Rare-facies failure in geographic federated learning is *bistable*, not
> gradual — and part of it is reversible at inference, for free.**

Across 45 training runs the rare facies either gets learned (~0.20–0.30 IoU) or
collapses to **exactly 0.000**. Only 3 runs land anywhere in between. Same code,
same data, different seed, opposite outcome.

That single fact reorganizes the whole talk:

1. **It explains the paper's headline failure.** "Class 5 → 0.0 IoU" is not a
   model that learned the class poorly. It is a model that fell into the wrong
   basin. Averaging weights across clients that mostly lack the class makes the
   bad basin much more likely.
2. **It explains why our interventions "worked."** Seed ensembling gains most of
   what we measured — because it buys more lottery tickets, not because it
   improves any single model.
3. **It explains why our own numbers were inflated.** Picking each run's best
   round on test data flatters a bistable process, and flatters unstable
   configurations *most*.
4. **It is actionable.** Logit adjustment at inference can pull a collapsed
   checkpoint partway back: **class-5 IoU 0.008 → 0.060 (7.5×) with τ=1.5, at
   zero training cost, and mean IoU improves slightly too.**

**One-sentence version for the talk:** *"The rare facies doesn't degrade — it
falls off a cliff, and which side of the cliff you land on is the seed. Some of
the fall is recoverable after training."*

---

## 1. What the accepted paper already says

Audience's starting point — summarize fast, don't re-derive.

| Claim | Number (as published, best-round) |
|---|---|
| IID FL ≈ centralized | Parihaka 0.686 vs 0.693; F3 0.787 vs 0.786 |
| Geographic non-IID degrades | mIoU −18–24% relative to centralized |
| Rare-class collapse | Class 5 → **0.0 IoU** at 5+ clients |
| Best simple mitigation | Client subsampling, +10.7% relative |
| Alternatives underperform | FedProx / FedBN degrade 1–12%; FedAvg wins |

**Published conclusion:** dominant failure mode is **data absence**, not
optimization dynamics. Ends on *"future work should explore knowledge
distillation and cross-client feature sharing."*

*(These are the published best-round numbers. Do not re-derive them on stage —
they are the audience's starting point. The corrected convention applies to
everything in §2 onward.)*

---

## 2. New since submission

### 2.1 Bistability — the headline (see above)
Figure: `figures/slides/bistability.png` (45 real runs; 29 dead / 13 learned /
3 between).

### 2.2 Recovery at inference — logit adjustment
`logits − τ·log(prior)` before the argmax. Three lines, no retraining, any
checkpoint. **State-dependent**: τ≈0.5 for a majority-biased checkpoint,
τ≈1.5 to rescue a collapsed one, τ=0 when already rare-biased. Pick τ on a
held-out crossline at ~zero cost.
Figures: `rescue.png`, `tau.png`.

### 2.3 THE RESULTS TABLE — use these numbers

Single models, **final round**, seeds {42, 123, 7, 99, 2025}:

| Config | n | mIoU | C5 recovered |
|---|---|---|---|
| GroupNorm | 4 | 0.4328 ± 0.0185 | 0/4 |
| Plain FedAvg (control) | 5 | 0.5326 ± 0.0309 | 0/5 |
| V3+frc (old "best") | 5 | 0.5421 ± 0.0753 | 2/5 |
| Baseline (rare agg) | 5 | 0.5514 ± 0.0121 | 1/5 |
| **T alone — best single model** | 5 | **0.5677 ± 0.0171** | 2/5 |

The ladder, end to end:

| Step | mIoU |
|---|---|
| Plain FedAvg | 0.5326 |
| + rare-class aggregation | 0.5514 |
| + best aggregation (T) | 0.5677 |
| **+ ensemble + logit adjustment** | **0.5965** |
| Centralized (late-epoch mean) | ~0.62 |

**+0.064 from plain FedAvg to the final method — ~73% of the gap to
centralized.** (The old best-round framing claimed 39%; correcting the *ceiling*
helped more than correcting our own numbers hurt.)

Ensemble detail (5 seeds, softmax average, final-round checkpoints):
τ=1.5 → **0.5965**; τ=0 → 0.5867. Best-round equivalent was 0.6155 at τ=0.5.
Source: `paper_figures/ensemble_v3_frc_{best,final}.txt`.

**Note the τ shift, 0.5 → 1.5.** This *confirms* the state-dependence theory
(§2.2): final-round checkpoints are more collapsed, so they need a stronger
correction. The theory predicted this before we measured it. Good Q&A material.

### 2.4 Three corrections to our own results ⚠️

Volunteer these. Each is a slide-worthy honesty beat, and (c) is a genuine
scientific result.

**(a) Checkpoint selection leaked test data.** We saved the best round by *test*
mIoU and reported that maximum. Optimistic — and unequally so:

| Config | Best-round | Final-round | Bias |
|---|---|---|---|
| Baseline (rare agg) | 0.5695 ± 0.0104 | 0.5514 ± 0.0121 | +0.018 |
| V3+frc ("our best") | 0.5849 ± 0.0085 | **0.5421 ± 0.0753** | **+0.043** |
| T alone | 0.5807 ± 0.0078 | **0.5677 ± 0.0171** | +0.013 |
| **Centralized Parihaka** | **0.6927 (ep45)** | 0.5853 (ep60) | **+0.107** |

Two claims flip: our "most stable" config is the *least* stable (seed 2025 ends
at 0.3919), and the simpler T variant wins on the unbiased metric.
Figure: `selection_bias.png`.

**The centralized ceiling was the worst offender.** 0.693 is a maximum over 20
test evaluations of a curve that oscillates 0.59–0.69 for all 60 epochs without
converging. Final-epoch (0.5853) is an equally arbitrary single sample near the
bottom. **Quote ~0.62, the late-epoch mean.** Do not quote 0.693 as a clean
upper bound. *(Note: the centralized model keeps C5 alive throughout,
0.26–0.51 — it never collapses. That, not the mIoU gap, is the sharpest
statement of the federated failure.)*

**(b) Two different mIoU definitions were in use.** `train_federated.py` uses
macro; `algorithms/fedseismic/metrics.py` uses `average='weighted'`, which is
dominated by C0/C2 (~77% of pixels) and can look healthy while C5 is at zero.
Report macro; label any weighted number explicitly.

**(c) Aggregation buys *probability of recovery*, not accuracy.** ★ *Reframed
2026-08-03 — this is now a finding, not just a correction.*

Against a proper `equal`-weight control, the mIoU differences between
aggregation variants are **within noise**. But the rare-class recovery rate is
not:

| Config | mIoU (final) | C5 recovered |
|---|---|---|
| Plain FedAvg (control) | 0.5326 ± 0.0309 | **0/5** |
| Rare-class aggregation | 0.5514 ± 0.0121 | **1/5** |
| T alone | 0.5677 ± 0.0171 | **2/5** |
| V3+frc | 0.5421 ± 0.0753 | **2/5** |

Plain FedAvg **never** recovers the rare facies. Aggregation roughly doubles the
odds. That ties directly to bistability: these methods shift the *probability*
of landing in the good basin — they do not make any single model better.

This is a stronger and more interesting claim than the old "+0.015 mIoU", and
it is the honest reading of the data.

**Caveat to state:** n=4–5 per config. The mIoU differences between aggregation
variants are within noise, and the C5 counts (0/5 vs 2/5) are suggestive, not
conclusive. Say so if pressed.

### 2.5 Normalization is NOT the cause ★ *New, 2026-08-03 — negative result*

Hypothesis tested: BatchNorm keeps per-client running statistics that get
averaged across clients — a documented non-IID failure mode, and a candidate
explanation for the paper's own FedBN null result. Swapped to GroupNorm,
everything else identical.

| Norm | mIoU (final) | C5 recovered |
|---|---|---|
| BatchNorm (control) | **0.5326 ± 0.0309** (n=5) | 0/5 |
| GroupNorm | **0.4328 ± 0.0185** (n=4) | 0/4 |

**GroupNorm is ~0.10 mIoU worse**, consistently — and it does not rescue the
rare class either. (Best-round numbers show the same gap: 0.5688 vs 0.4708,
with no overlap between the two sets of seeds.)

*Why:* the FL BatchNorm literature concerns classification with skewed label
distributions. Here every client holds slices of **one survey**, so the
low-level input statistics (amplitude, texture, frequency) are shared. BN's
per-channel statistics help; GroupNorm discards that for nothing. The non-IID-ness
is in *which facies appear*, not in the input distribution.

**Talk value:** this is a direct, strong answer to "why did FedBN fail?" — we
tested the strongest version of the normalization hypothesis and it is not the
lever. Pairs naturally with the honesty thread.

---

## 3. Talk arc (15–17 min)

| # | Beat | Slide |
|---|---|---|
| 1 | Train together, share no data | statement |
| 2 | Real partners hold real geography | `gap.png` |
| 3 | One facies went to zero — not degraded, absent | statement |
| 4 | The cause is absence, not drift | statement |
| 5 | **Section: what we found since** | section |
| 6 | **Bistable, not gradual** ← the finding | `bistability.png` |
| 7 | Recovering the gap step by step | `ladder.png` |
| 8 | Subtract the class prior — 3 lines | statement |
| 9 | **A collapsed facies, brought back** ← money slide | `rescue.png` |
| 10 | How hard to push depends on the model | `tau.png` |
| 11 | Where we had been fooling ourselves | `selection_bias.png` |
| 12 | Normalization is not the cause (negative result) | `norm.png` |
| 13 | Three takeaways | bullets |
| 14 | Q&A | section |

Deck: `IMAGE2026_FL_Seismic.pptx`, rebuilt by `build_presentation.py`.
Charts: `figures/slide_charts.py` → `figures/slides/`.

---

## 4. Caveats to state before being asked

- **Small n.** 4–5 seeds per config. mIoU differences between aggregation
  variants are within noise; the C5 recovery counts are suggestive, not
  conclusive. Say this plainly if pressed on significance.
- **Recall-per-slice's mIoU gain is within noise** (+0.0004, n=3). It is a
  *rare-class* contribution (max-C5 +0.011, best final C5 0.280), not an mIoU one.
- **Ensembling costs 5–10× training.** Logit adjustment does not — say so.
- **All new results are Parihaka**, 20 clients, sr=0.25. F3 untested.
- **τ needs a held-out slice.** Zero *training* cost, not zero cost.
- **The 0.6176 ten-model number is not recomputed.** Only the 5-seed V3+frc
  ensemble was redone on final-round checkpoints (0.5965). Do not quote 0.6176.

---

## 5. Open items

- [ ] **Update the deck to the corrected numbers** — `ladder.png` needs the
      0.5326 → 0.5965 ladder and the ~0.62 ceiling; `build_presentation.py`
      still narrates 0.6155 and "39% of the gap".
- [ ] Swap the ablation ladder's "best config" from V3+frc to **T alone**.
- [ ] Confirm author list / affiliations on the title slide.
- [ ] Add OLIVES + GT logos → `figures/logos/{olives,gatech}.png`, then rebuild.
- [ ] Back up `logs_*/`, `results/` locally — sole provenance, one machine.
      *(Decided: these stay off the remote.)*
- [ ] Optional: EMA of the global model (~5 lines) — may damp bistability far
      more cheaply than 5–10× ensembling. Best remaining idea in `plan.md`.
- [ ] Deferred until after IMAGE: refactor/organization (see `plan.md` and the
      code-review findings). Not before the talk.
