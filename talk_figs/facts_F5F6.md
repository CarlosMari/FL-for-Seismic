

<!-- BEGIN F6 -->
## F6 - head-update x-ray (presence + ranking only)

Run paths (local repo tree; `head_outc/` + `labels/` live here, the full
`view/`+`received/` logs live on the HDD):

- `/home/carlos/projects/FL-for-Seismic/results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n5_batchnorm/seed_0`  (n5, N=5, fedavg batchnorm, seed_0)
- `/home/carlos/projects/FL-for-Seismic/results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0`  (n3, N=3, fedavg batchnorm, seed_0)

**Rounds used: round 020 ONLY.** `head_outc/` holds the recovered
outc (segmentation head) upload-minus-received delta for the last round
only (rounds 1-19 were unrecoverable - see `seismic_channel.py` header).
The requested rounds 1-10 average is therefore NOT available; the figure
and every number below use round 020 and are labelled as such.


### n5 (N=5) - `f6_xray_n5.png/.pdf/.svg`

- facies order: ['Upper North Sea', 'Middle North Sea', 'Lower North Sea', 'Rijnland/Chalk', 'Scruff', 'Zechstein'] (rare classes [4, 5])
- absent (site, facies) pairs: 3 -> [('Operator A', 'Zechstein'), ('Operator B', 'Zechstein'), ('Operator C', 'Zechstein')]
- 'everything present' base-rate accuracy: 0.900
- presence balanced accuracy from the round-020 bias change (3 documented threshold rules): range_midpoint = 0.796, classmeans_midpoint = 0.796, sign0 = 0.722
- within-facies Pearson (bias, log counts, seed_0): +0.763
- within-facies Pearson (row norms, seed_0): -0.585
- within-column ranks shown in panel A (centered bias) rows 1..5, panel B (true share) rows 1..5; absent cells marked '—' with a bold outline
- saved figure aspect 1.90:1 (slide placeholder ASPECT['f6'] = 1.9)

### n3 (N=3) - `f6_xray_n3.png/.pdf/.svg`

- facies order: ['Upper North Sea', 'Middle North Sea', 'Lower North Sea', 'Rijnland/Chalk', 'Scruff', 'Zechstein'] (rare classes [4, 5])
- absent (site, facies) pairs: 2 -> [('Operator A', 'Zechstein'), ('Operator B', 'Zechstein')]
- 'everything present' base-rate accuracy: 0.889
- presence balanced accuracy from the round-020 bias change (3 documented threshold rules): range_midpoint = 0.750, classmeans_midpoint = 0.719, sign0 = 0.781
- within-facies Pearson (bias, log counts, seed_0): +0.438
- within-facies Pearson (row norms, seed_0): -0.336
- within-column ranks shown in panel A (centered bias) rows 1..3, panel B (true share) rows 1..3; absent cells marked '—' with a bold outline
- saved figure aspect 1.90:1 (slide placeholder ASPECT['f6'] = 1.9)

### Pooled presence check (all 8 jobs x 3 seeds, each N)

- N=3: n=216 (192 present / 24 absent), balanced accuracy pooled over all runs range_midpoint = 0.693, classmeans_midpoint = 0.729, sign0 = 0.724; per-run mean over the 12 runs range_midpoint = 0.815, classmeans_midpoint = 0.896, sign0 = 0.724; base rate 0.889
- N=5: n=360 (324 present / 36 absent), balanced accuracy pooled over all runs range_midpoint = 0.767, classmeans_midpoint = 0.767, sign0 = 0.753; per-run mean over the 12 runs range_midpoint = 0.861, classmeans_midpoint = 0.955, sign0 = 0.753; base rate 0.900

### Per-job pooled within-facies Pearson (all 8 jobs)

- fedavg_noniid_a0.1_lNA_dna_n3_batchnorm: bias +0.512, row -0.332
- fedavg_noniid_a0.1_lNA_dna_n3_groupnorm: bias +0.570, row -0.291
- fedavg_noniid_a0.1_lNA_dna_n5_batchnorm: bias +0.615, row -0.407
- fedavg_noniid_a0.1_lNA_dna_n5_groupnorm: bias +0.642, row -0.371
- fedbn_noniid_a0.1_lNA_dna_n3_batchnorm: bias +0.503, row -0.332
- fedbn_noniid_a0.1_lNA_dna_n3_groupnorm: bias +0.540, row -0.248
- fedbn_noniid_a0.1_lNA_dna_n5_batchnorm: bias +0.361, row -0.408
- fedbn_noniid_a0.1_lNA_dna_n5_groupnorm: bias +0.570, row -0.462

### Comparison to findings.md

- findings.md line 572: within-class bias Pearson **+0.36 to +0.64 on all 8
  jobs**. Reproduced: +0.361 to +0.642 per job (same aggregation as
  `score_job`). Match within 0.01 - no flag.
- findings.md line 580: bias balanced accuracy (midpoint threshold)
  **0.76 at N=3, 0.90 at N=5**, against a 0.89-0.90 base rate.
  The exact historical threshold rule is not recorded anywhere in the
  repo ('midpoint' is ambiguous), so three defensible rules are reported
  above (range midpoint, class-means midpoint, sign). None of them
  reproduces BOTH findings values within 0.02:
    - N=3 pooled range_midpoint = 0.693 vs 0.76 (|diff| 0.067 - **FLAG: mismatch 0.067 > 0.02**)
    - N=3 pooled classmeans_midpoint = 0.729 vs 0.76 (|diff| 0.031 - **FLAG: mismatch 0.031 > 0.02**)
    - N=3 pooled sign0 = 0.724 vs 0.76 (|diff| 0.036 - **FLAG: mismatch 0.036 > 0.02**)
    - N=3 per-run mean range_midpoint = 0.815 vs 0.76 (|diff| 0.055 - **FLAG: mismatch 0.055 > 0.02**)
    - N=3 per-run mean classmeans_midpoint = 0.896 vs 0.76 (|diff| 0.136 - **FLAG: mismatch 0.136 > 0.02**)
    - N=3 per-run mean sign0 = 0.724 vs 0.76 (|diff| 0.036 - **FLAG: mismatch 0.036 > 0.02**)
    - N=5 pooled range_midpoint = 0.767 vs 0.90 (|diff| 0.133 - **FLAG: mismatch 0.133 > 0.02**)
    - N=5 pooled classmeans_midpoint = 0.767 vs 0.90 (|diff| 0.133 - **FLAG: mismatch 0.133 > 0.02**)
    - N=5 pooled sign0 = 0.753 vs 0.90 (|diff| 0.147 - **FLAG: mismatch 0.147 > 0.02**)
    - N=5 per-run mean range_midpoint = 0.861 vs 0.90 (|diff| 0.039 - **FLAG: mismatch 0.039 > 0.02**)
    - N=5 per-run mean classmeans_midpoint = 0.955 vs 0.90 (|diff| 0.055 - **FLAG: mismatch 0.055 > 0.02**)
    - N=5 per-run mean sign0 = 0.753 vs 0.90 (|diff| 0.147 - **FLAG: mismatch 0.147 > 0.02**)
  The direction and the conclusion (well above the 0.5 chance of a
  balanced score, well below the ~0.95 AUC gloss findings.md retired)
  hold under every rule. No single rule reproduces the (0.76, 0.90)
  pair, so the historical numbers are treated as not exactly
  reproducible and the rule set above stands as the honest record.
- findings.md line 580: presence pair counts 48+/6- at N=3, 81+/9- at N=5,
  'every negative is c5'. Reproduced exactly: 2 absent pairs per N=3 seed
  and 3 per N=5 seed, all (site, Zechstein); seed_0 shown in the figures
  has 2 (N=3) and 3 (N=5) absent pairs. No flag.
- Wording: this figure shows **presence and within-facies ranking only**.
  Nothing here is an 'estimated amount'; exact counts did not calibrate on
  F3 (findings.md 2026-10-16: 12-shadow pilot closed, no transferable slope).
<!-- END F6 -->





<!-- BEGIN F5 -->
## F5 - the fingerprint (site re-identification from updates)

Input runs (HDD; `view/` holds `delta_proj`, `last_*` and `bn::*` per upload):

- `f5a`: `/mnt/storage/results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm` (N=3, seeds ['seed_0', 'seed_1', 'seed_42'])
- `f5b`: `/mnt/storage/results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_groupnorm` (N=3, seeds ['seed_0', 'seed_1', 'seed_42'])
- `f5a_n5`: `/mnt/storage/results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n5_batchnorm` (N=5, seeds ['seed_0', 'seed_1', 'seed_42'])
- `f5b_n5`: `/mnt/storage/results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n5_groupnorm` (N=5, seeds ['seed_0', 'seed_1', 'seed_42'])

Rounds used: 1-20 (all logged rounds; full participation). Feature centering is exactly `bn_site_auc`: per (round, layer) subtract the across-site mean, z-score each channel, concatenate layers. Panel B uses `delta_proj` (4096-d full-update sketch) with the same per-round centering.

Metric: leave-one-round-out cosine nearest-neighbor **top-1 site match** (the neighbor of each upload must come from a different round); chance = 1/N exactly under equal participation.

### f5a - `f5a_fingerprint_bn.mp4/.gif` + `f5a_final.png` still

- top-1 NN (all layers): seed_0 0.967, seed_1 1.000, seed_42 1.000 -> 3-seed mean 0.989 (chance 0.33)
- top-1 NN (first layer only): seed_0 0.950, seed_1 1.000, seed_42 0.983 -> 3-seed mean 0.978 (chance 0.33)
- layers used (all): 18 running_mean tensors (down1.maxpool_conv.1.double_conv.1 ... up4.conv.double_conv.4); first-layer-only variant uses `down1.maxpool_conv.1.double_conv.1` - that is `sorted(bn_keys)[0]`, the same split `bn_site_auc` calls 'first-encoder-layer'.
- seed selection: separations (top-1) seed_0 0.967, seed_1 1.000, seed_42 1.000, median 1.000; margins seed_0 0.288, seed_1 0.280, seed_42 0.339, median 0.288. Animated seed: **seed_1** (closest to the 3-seed median, ties broken by margin then seed order).

### f5b - `f5b_fingerprint_gn.mp4/.gif` + `f5b_final.png` still

- top-1 NN (all layers): seed_0 1.000, seed_1 0.983, seed_42 1.000 -> 3-seed mean 0.994 (chance 0.33)
- top-1 NN (first layer only): seed_0 1.000, seed_1 0.983, seed_42 1.000 -> 3-seed mean 0.994 (chance 0.33)
- layers: `delta_proj` is a single 4096-d sketch of the full update with no layer structure, so the first-layer-only variant coincides with the all-layers variant.
- seed selection: separations (top-1) seed_0 1.000, seed_1 0.983, seed_42 1.000, median 1.000; margins seed_0 0.240, seed_1 0.224, seed_42 0.272, median 0.240. Animated seed: **seed_0** (closest to the 3-seed median, ties broken by margin then seed order).

### f5a_n5 - `f5a_n5_fingerprint_bn.mp4/.gif` + `f5a_n5_final.png` still

- top-1 NN (all layers): seed_0 0.940, seed_1 0.940, seed_42 0.940 -> 3-seed mean 0.940 (chance 0.20)
- top-1 NN (first layer only): seed_0 0.940, seed_1 0.950, seed_42 0.910 -> 3-seed mean 0.933 (chance 0.20)
- layers used (all): 18 running_mean tensors (down1.maxpool_conv.1.double_conv.1 ... up4.conv.double_conv.4); first-layer-only variant uses `down1.maxpool_conv.1.double_conv.1` - that is `sorted(bn_keys)[0]`, the same split `bn_site_auc` calls 'first-encoder-layer'.
- seed selection: separations (top-1) seed_0 0.940, seed_1 0.940, seed_42 0.940, median 0.940; margins seed_0 0.272, seed_1 0.267, seed_42 0.237, median 0.267. Animated seed: **seed_1** (closest to the 3-seed median, ties broken by margin then seed order).

### f5b_n5 - `f5b_n5_fingerprint_gn.mp4/.gif` + `f5b_n5_final.png` still

- top-1 NN (all layers): seed_0 0.940, seed_1 0.940, seed_42 0.950 -> 3-seed mean 0.943 (chance 0.20)
- top-1 NN (first layer only): seed_0 0.940, seed_1 0.940, seed_42 0.950 -> 3-seed mean 0.943 (chance 0.20)
- layers: `delta_proj` is a single 4096-d sketch of the full update with no layer structure, so the first-layer-only variant coincides with the all-layers variant.
- seed selection: separations (top-1) seed_0 0.940, seed_1 0.940, seed_42 0.950, median 0.940; margins seed_0 0.251, seed_1 0.266, seed_42 0.276, median 0.266. Animated seed: **seed_1** (closest to the 3-seed median, ties broken by margin then seed order).

### Which projection did the animation use

The headline animations are the N=3 pair (`f5a` BN vs `f5b` GN) so the side-by-side compares the two normalization schemes at the same cohort size; the N=5 variants (`f5a_n5`, `f5b_n5`) are extras. Plain PCA (fit once on all rounds of the chosen run) was used in every case - no UMAP - and the all-layers feature was shown; the first-layer-only top-1 is recorded above as the fallback.
- f5a: PC1+PC2 explain 19%+9% of centered variance; the three site clusters separate cleanly in 2-D.
- f5b: PC1+PC2 explain 12%+7% of centered variance; the three site clusters separate cleanly in 2-D.
- f5a_n5: PC1+PC2 explain 9%+8% of centered variance; the five site groups are visibly more interleaved than at N=3 (a few A/D/B points overlap in 2-D) but each site still owns a distinct region; this panel is an extra, and the re-identification metric does not depend on the 2-D projection.
- f5b_n5: PC1+PC2 explain 7%+7% of centered variance; the five site groups are visibly more interleaved than at N=3 (a few A/D/B points overlap in 2-D) but each site still owns a distinct region; this panel is an extra, and the re-identification metric does not depend on the 2-D projection.

Note on the word 'first-encoder-layer' (BatchNorm panels): `bn_site_auc` builds its 'first' split from `sorted(bn_keys)[0]`, which on this UNet is `down1.maxpool_conv.1.double_conv.1` (the first *down* block), not `inc.double_conv.1`. We follow the repo definition exactly and name the tensor above so the reader can see which layer it is.

### Comparison to findings.md

- findings.md line 580: 'BN top-1 nearest-neighbor match across rounds is 0.98 vs 0.33 chance at N=3 and 0.94 vs 0.20 at N=5'. Reproduced: BN N=3 3-seed mean 0.989 vs chance 0.33 (|diff| 0.009 - no flag); BN N=5 mean 0.940 vs 0.20 chance (|diff| 0.000 - no flag).
- findings.md line 405: GroupNorm 'full-update sketch 0.93-0.98', GN affine weights 0.97/1.00. Our GroupNorm channel is the full-update sketch (`delta_proj`): N=3 per-seed 1.000, 0.983, 1.000, N=5 per-seed 0.940, 0.940, 0.950. The N=5 values sit in [0.93, 0.98]; the N=3 values reach 1.000, i.e. above the quoted 0.98 ceiling by 0.02 - borderline, flagged as a gloss difference (different seeds/definition of 'sketch' aggregation cannot be re-derived from the repo). The substance - sites stay identifiable without statistics - reproduces.
- No input was missing for F5: all 12 runs (4 jobs x 3 seeds) have complete `view/` logs on the HDD with `bn::` keys (batchnorm jobs) and `delta_proj` (every job).
<!-- END F5 -->





