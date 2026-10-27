# F1/F2 facts (sponsor talk)

- Run: `results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0` (fedavg, noniid contiguous blocks, a=0.1, n=3, batchnorm), seed 0.
- Seed: 0 (all splits and crossline choices are deterministic in this seed).
- Deployed models: `local/client_000.pt` - `local/client_002.pt` (each site's own BN); official test row uses `global_final.pt`.
- Metrics quoted are present-class mIoU (`fedseismic.eval.metrics.evaluate_loader`).

## F1 - survey split into operator blocks
### N=3 split (results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0)

| operator | crosslines | block | train | holdout tail | facies mix (train pixels, 0..5) | lacks Zechstein |
|---|---|---|---|---|---|---|
| Operator A | 233 | 0-232 | 0-185 | 186-232 | 26.0% 13.0% 46.3% 9.3% 5.4% 0.0% | yes (0 class-5 px) |
| Operator B | 233 | 233-465 | 233-418 | 419-465 | 31.3% 12.5% 47.4% 6.0% 2.8% 0.0% | in train split |
| Operator C | 235 | 466-700 | 466-653 | 654-700 | 26.3% 10.2% 50.5% 6.1% 1.6% 5.4% | no |

- Operators with **zero** Zechstein (class 5) pixels anywhere in their block: **1 of 3** (operator A).
- Operators with zero Zechstein in their *training* split (the pi histograms): **2 of 3** (operator A, B).
- Note: operator B's only Zechstein pixels (0.19% of its block) lie in its holdout tail (xline 419-465); its training split has none.
### N=5 split (results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n5_batchnorm/seed_0)

| operator | crosslines | block | train | holdout tail | facies mix (train pixels, 0..5) | lacks Zechstein |
|---|---|---|---|---|---|---|
| Operator A | 140 | 0-139 | 0-111 | 112-139 | 25.7% 13.4% 47.7% 9.5% 3.7% 0.0% | yes (0 class-5 px) |
| Operator B | 140 | 140-279 | 140-251 | 252-279 | 28.0% 12.2% 43.6% 8.5% 7.8% 0.0% | yes (0 class-5 px) |
| Operator C | 140 | 280-419 | 280-391 | 392-419 | 31.7% 12.6% 48.0% 5.5% 2.2% 0.0% | yes (0 class-5 px) |
| Operator D | 140 | 420-559 | 420-531 | 532-559 | 27.6% 10.7% 50.8% 6.4% 2.0% 2.5% | no |
| Operator E | 141 | 560-700 | 560-672 | 673-700 | 27.0% 10.6% 52.2% 4.2% 1.3% 4.7% | no |

- Operators with **zero** Zechstein (class 5) pixels anywhere in their block: **3 of 5** (operator A, B, C).
- Operators with zero Zechstein in their *training* split (the pi histograms): **3 of 5** (operator A, B, C).

## F2 - section triptych (chosen crosslines)

Representative crossline = median per-section error (1 - present-class mIoU) of the row's own model over its whole holdout - not the best section. Global row: median over all 701 crosslines of test1.

| row | crossline | why | section mIoU | holdout mIoU |
|---|---|---|---|---|
| Operator A (client 0, local/client_000.pt) | 196 | median error of 47 holdout sections (err median 0.030, range 0.017-0.053) | 0.970 | 0.968 |
| Operator B (client 1, local/client_001.pt) | 464 | median error of 47 holdout sections (err median 0.293, range 0.044-0.314) | 0.707 | 0.721 |
| Operator C (client 2, local/client_002.pt) | 696 | median error of 47 holdout sections (err median 0.040, range 0.028-0.228) | 0.961 | 0.791 |
| Official test (global_final.pt) | 256 | median error of 701 test1 sections | 0.743 | 0.588 (all test1) |

- Global model on the official test volume: present-class mIoU = 0.588; per-class IoU 0.853 0.765 0.893 0.527 0.489 0.000. Zechstein (class 5) is present in the official test but never recovered (IoU 0.000).
- The run's logged score (results.json `final` = 0.568) is the mean of the test1 and test2 evaluations; with the saved `global_final.pt`, test1 = 0.588 and test2 = 0.547 (mean 0.568).
- Talk hook: operator B's chosen section (xline 464) is a salt diapir with a Zechstein core - B's only Zechstein lives in its holdout tail - and the deployed model calls it Scruff/Rijnland (see the vermillion error blob).
- Figure aspect: f2_triptych_n3 = 1.46:1 (4 rows x 4 columns of true-aspect 401x255 panels; a 3:1 overall figure is not geometrically possible without squashing the seismic).
