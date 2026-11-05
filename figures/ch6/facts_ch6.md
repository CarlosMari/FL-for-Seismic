# Chapter 6 figure facts

Generated 2 Nov 2026 on `talk-figures` (HEAD `494aa64`, uncommitted). CPU inference and rescoring only; no training, and the GPU stayed idle. Machine-readable values are in `figures/ch6/data/`.

No Chapter 6 source file exists in the repo, so "quoted" values are those listed in the figure brief. Table 6.2 could not be compared cell by cell; all eight recomputed rows are listed under Fig 6.3 for that check.

Rule: `FLAG` means a recomputed value differs from the quoted value by more than 0.01.

**FLAG count: 3, all in Fig 6.6.**

## Fig 6.1, `fig6_1_f3_sites.pdf`

- Run: `results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0`; the N=5 check uses the `n5` sibling. Seed 0. Facies mix from `labels/pi.json`, and it matches a direct count on the train label cube.
- Sections: the median-error holdout crosslines from Fig 6.2 (196, 464, 696).
- Splits: A 0-185 / 186-232; B 233-418 / 419-465; C 466-653 / 654-700.

| Quoted | Recomputed | Status |
|---|---|---|
| Lower North Sea 46.3 / 47.4 / 50.5 % | 46.3 / 47.4 / 50.5 | OK |
| Scruff 5.4 / 2.8 / 1.6 % | 5.4 / 2.8 / 1.6 | OK |
| Zechstein 0 / 0 / 5.4 % of training pixels | 0 / 0 / 5.4 | OK |
| Site B's Zechstein only in holdout, 419-465 | 0 train pixels, 44,629 holdout pixels | OK |
| N=5: three of five sites have no Zechstein | A, B, C have 0 in train and holdout | OK |

## Fig 6.2, `fig6_2_triptych.pdf`

- Same run, seed 0, round 20. Sites use `local/client_00{0,1,2}.pt`; test1 uses `global_final.pt`. Present-class mIoU (`evaluate_loader`).
- Median-error crosslines: A 196, B 464, C 696, test1 256. Section mIoU 0.970, 0.707, 0.961, 0.743.
- Height 4.97 in. Site B's arrow points to the centroid of the 2,173 Zechstein pixels in crossline 464.
- Site C's section score (0.961) is far above its holdout score (0.791): the median section is much better than the pooled score, partly because Scruff IoU pooled over the holdout is 0.098.

| Quoted | Recomputed | Status |
|---|---|---|
| Holdout mIoU 0.968 / 0.721 / 0.791 | 0.968 / 0.721 / 0.791 | OK |
| Official test1 0.588 | 0.588 | OK |
| Per-class 0.853 0.765 0.893 0.527 0.489 0.000 | 0.853 0.765 0.893 0.528 0.489 0.000 | OK (0.001 on class 3) |

## Fig 6.3, `fig6_3_head_xray.pdf`

- Runs: `results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n{3,5}_batchnorm/seed_0`. **Round 20 only** (`head_outc/`).
- The pooled metrics use all eight `seismic_f3` jobs: four configurations at each N, three seeds each.
- X-axis labels abbreviate Upper, Middle and Lower North Sea to fit.
- The "training share" shade is scaled within each column and carries no facies colour, because it encodes magnitude.

| Quoted | Recomputed | Status |
|---|---|---|
| Absent pairs 2 (N=3), 3 (N=5), all Zechstein | 2 and 3, all Zechstein | OK |
| Sign-rule balanced accuracy 0.724 (216 pairs) | 0.724 (216 pairs, TPR 0.448, TNR 1.000) | OK |
| Sign-rule balanced accuracy 0.753 (360 pairs) | 0.753 (360 pairs, TPR 0.506, TNR 1.000) | OK |
| Base rates 0.889 / 0.900 | 0.889 / 0.900 | OK |
| Within-facies bias +0.36 to +0.64 | +0.361 to +0.642 | OK |
| Within-facies row -0.25 to -0.46 | -0.248 to -0.462 | OK |

Per-job within-facies Pearson, for checking Table 6.2:

| Job | Bias | Row |
|---|---|---|
| fedavg n3 batchnorm | +0.512 | -0.332 |
| fedavg n3 groupnorm | +0.570 | -0.291 |
| fedavg n5 batchnorm | +0.615 | -0.407 |
| fedavg n5 groupnorm | +0.642 | -0.371 |
| fedbn n3 batchnorm | +0.503 | -0.332 |
| fedbn n3 groupnorm | +0.540 | -0.248 |
| fedbn n5 batchnorm | +0.361 | -0.408 |
| fedbn n5 groupnorm | +0.570 | -0.462 |

## Fig 6.4, `fig6_4_fingerprint.pdf`

- Runs: `/mnt/storage/results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n{3,5}_{batchnorm,groupnorm}`, seeds 0, 1, 42, rounds 1-20. Panels show N=3.
- Plotted seeds, from `pick_seed`: **BN seed 1, GN seed 0**. Their top-1 is 1.000, the 3-seed median. Each panel prints this and the 3-seed mean.
- Per-seed N=3: BN 0.967, 1.000, 1.000; GN 1.000, 0.983, 1.000.
- PCA variance: BN 19% + 9%; GN 12% + 7%.
- First layer: `down1.maxpool_conv.1.double_conv.1`, as in `bn_site_auc`.

| Quoted | Recomputed 3-seed mean | Status |
|---|---|---|
| BN top-1 0.989 / 0.940 | 0.989 / 0.940 | OK |
| First layer 0.978 / 0.933 | 0.978 / 0.933 | OK |
| GN sketch 0.994 / 0.943 | 0.994 / 0.943 | OK |
| Chance 0.33 / 0.20 | 1/3, 1/5 | OK |

## Fig 6.5, `fig6_5_bn_blackout.pdf`

- Runs: `results/runs/multisurvey/fedavg_noniid_a0.1_lNA_dna_n4/seed_0`, `results/runs/multisurvey_fedbn/fedbn_noniid_a0.1_lNA_dna_n4/seed_0`. Seed 0, final round.
- **Orientation:** Parihaka here used `parihaka_data_train.npz`, the original orientation with reflectors vertical. It is rotated with `np.rot90` for display only; inference used the training orientation.
- BN re-estimation: one train-mode pass over each site's training loader without gradients (70, 71, 79, 79 batches).
- Crosslines, median error under (b): Parihaka client 2 **346** of 78; F3 client 0 **336** of 70. Section mIoU under (b) 0.793 and 0.844.
- Panel corners show the **whole-holdout** masked mIoU, matching the quoted numbers.
- **Not reproducible to the crossline:** the re-estimate depends on the loader's shuffle, so scores move in the third decimal and the median crossline moves. The talk run gave crosslines 341 and 313, and F3 c0 (b) 0.8546. Here it is 0.8573, which prints as 0.86 in the panel against the chapter's 0.85. That is within the 0.01 rule, so it is not flagged.
- Parihaka and F3 share palette colours by class index. They are different facies, so the figure has one legend row per survey.

| Quoted, masked | Recomputed | Status |
|---|---|---|
| Parihaka (a) 0.17 / 0.15 | 0.166 / 0.152 | OK |
| Parihaka (b) 0.79 / 0.64 | 0.793 / 0.637 | OK |
| Parihaka (c) 0.73 / 0.81 | 0.727 / 0.810 | OK |
| F3 (a) 0.32 / 0.29 | 0.324 / 0.288 | OK |
| F3 (b) 0.85 / 0.52 | 0.857 / 0.522 | OK (see above) |
| F3 (c) 0.92 / 0.44 | 0.917 / 0.436 | OK |
| Unmasked (a) Parihaka 0.0003 / 0.0004 | 0.0003 / 0.0004 | OK |

## Fig 6.6, `fig6_6_three_surveys.pdf`

- Script: `talk_figs/thesis_ch6_rescore_k3.py`, not `compare_multisurvey3_k3.py`. The comparison script needs `results.json`, so it would silently drop the Parihaka solo, and its tables were not present. My script calls the same `rescore_run` and writes to `figures/ch6/data/`.
- Seed 0, buffer 50, Parihaka `_rot90`, masked argmax on the deployed personal states.
- Arms:
  - Solo F3: `results/runs/multisurvey3_k3_solo_a/fedpersurvey_noniid_a0.1_lNA_dna_n1/seed_0`, 20 rounds.
  - Solo Parihaka: `results/runs/multisurvey3_k3_solo_b/fedpersurvey_noniid_a0.1_lNA_dna_n1/seed_0`, **provisional, 19/20 rounds**. It has no `run_meta.json`; scored using the solo A config with `survey_solo="b"`, the only difference between those sweep files other than the output path.
  - Per-survey heads: `results/runs/multisurvey3_k3_fedpersurvey/fedpersurvey_noniid_a0.1_lNA_dna_n3/seed_0`, 20 rounds.
  - FedHCA2-style control: `results/runs/multisurvey3_k3_fedhca2_restart_s0/fedhca2_noniid_a0.1_lNA_dna_n3/seed_0`, 20 rounds.
  - Solo Penobscot: never run.
- The figure plots the recomputed values.

| Quoted, per-survey heads or solo | Recomputed present-class | Status |
|---|---|---|
| Per-survey heads F3 0.444 | 0.659 | **FLAG** 0.444 vs 0.659 |
| Per-survey heads Parihaka 0.288 | 0.446 | **FLAG** 0.288 vs 0.446 |
| Per-survey heads Penobscot 0.827 | 0.957 | **FLAG** 0.827 vs 0.957 |
| Solo F3 0.617 | 0.617 | OK |
| Solo Parihaka 0.645 | 0.645 | OK |

**Reason for the FLAGs:** the three quoted values are the FedHCA2-style control's scores (0.444, 0.288, 0.827), attributed to the wrong arm. Evidence:

- `results/tables/fedhca2_restart_s0_provisional.json` holds exactly those values.
- The per-survey run's own `results.json` has `local_mean` 0.6874 and `local_worst` 0.4462. These equal the mean and minimum of my 0.6587, 0.4463, 0.9573.
- `k3_fedpersurvey_s0_provisional.json` matches my rescore.

**Consequence:** the same three numbers appear in the 1 Nov decision entry in `findings.md` (lines 70-72). With the correct arm, per-survey heads are **above** solo on F3 (0.659 vs 0.617, +0.042) and still below on Parihaka (0.446 vs 0.645 provisional, -0.199). "Degraded F3" does not hold for this arm at seed 0. Neither file was edited.

### Answers

**(i) All-class or present-class.** Every class of each survey occurs in its holdout, so the survey's own all-class and present-class means are identical. The other all-class figure divides by the 19 union rows.

| Arm, survey | Present-class | Own-survey all-class | 19-row union |
|---|---|---|---|
| Solo F3 | 0.617 | 0.617 | 0.195 |
| Solo Parihaka | 0.645 | 0.645 | 0.204 |
| Per-survey heads F3 | 0.659 | 0.659 | 0.208 |
| Per-survey heads Parihaka | 0.446 | 0.446 | 0.141 |
| Per-survey heads Penobscot | 0.957 | 0.957 | 0.353 |
| FedHCA2-style F3 | 0.444 | 0.444 | 0.140 |
| FedHCA2-style Parihaka | 0.288 | 0.288 | 0.091 |
| FedHCA2-style Penobscot | 0.827 | 0.827 | 0.305 |

The five quoted values as written (0.444, 0.288, 0.827, 0.617, 0.645) are all present-class, equal to own-survey all-class.

**(ii) FedHCA2-style per survey, for the `[MISSING]` placeholders:** F3 0.444, Parihaka 0.288, Penobscot 0.827.

### Caveats for this comparison

- Single seed.
- The control's checkpoint is post-aggregation, while the per-survey personal state is the last locally trained one (`findings.md`, 30 Oct), which limits attribution to aggregation alone.
- `style.py` marks the Parihaka formation names as still to be confirmed against the challenge docs, and Penobscot uses placeholder names (Facies 1-7).
