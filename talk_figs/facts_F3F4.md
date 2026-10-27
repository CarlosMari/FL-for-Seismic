# Facts: F3 blackout + F4 training movie (sponsor talk)

## F3 - The blackout (averaged vs local BN)

- FedAvg run: `results/runs/multisurvey/fedavg_noniid_a0.1_lNA_dna_n4/seed_0` (seed 0, UNet(1,12,norm=batch,groups=8), `global_final.pt`)
- FedBN run: `results/runs/multisurvey_fedbn/fedbn_noniid_a0.1_lNA_dna_n4/seed_0` (deployed = `local/client_*.pt`)
- Parihaka = survey B, union class rows 6-11 (block (6,12)); F3 = rows 0-5 (block (0,6))
- Panel crossline: Parihaka client 2 holdout crossline 341 (median-error crossline under (b) out of 78 holdout crosslines; sample: xl 313: 0.800, xl 322: 0.802, xl 331: 0.788, xl 340: 0.790, xl 349: 0.791, xl 358: 0.790, xl 367: 0.789, xl 376: 0.790, xl 385: 0.797)
- F3 row crossline: client 0 holdout crossline 313 (median-error under (b))
- BN re-estimation: one full pass in train() mode + torch.no_grad() over the client's own train loader (batches: c0=70, c1=71, c2=79, c3=79), then eval()
- Device: cpu (GPU fully occupied by another job at run time)

### mIoU computed here (present-class, whole client holdout, IGNORE=255 dropped)

Two conventions are recorded because findings.md's 2026-10-19 numbers were computed with UNMASKED argmax (the run's own `results.json` `global_on_local` = 0.15303 = mean(0.3238, 0.2876, 0.0003, 0.0004) pins it exactly), while this figure task specifies the deployed masked argmax (`_score_deployed`). Masked is the primary (shown in the figures).

| model | survey | client | mIoU masked (deployed) | mIoU unmasked (findings conv.) |
|---|---|---|---|---|
| (a) | parihaka | 2 | 0.1656 | 0.0003 |
| (b) | parihaka | 2 | 0.7926 | 0.7796 |
| (d) | parihaka | 2 | 0.7271 | 0.7271 |
| (a) | parihaka | 3 | 0.1519 | 0.0004 |
| (b) | parihaka | 3 | 0.6365 | 0.6309 |
| (d) | parihaka | 3 | 0.8103 | 0.8103 |
| (a) | f3 | 0 | 0.3238 | 0.3238 |
| (b) | f3 | 0 | 0.8546 | 0.8540 |
| (d) | f3 | 0 | 0.9171 | 0.9171 |
| (a) | f3 | 1 | 0.2876 | 0.2876 |
| (b) | f3 | 1 | 0.5230 | 0.5238 |
| (d) | f3 | 1 | 0.4355 | 0.4355 |

### Comparison vs findings.md (2026-10-19 diagnostic; flag >0.02)

- (a) Parihaka client 2 (averaged BN): masked 0.1656 (delta 0.1653 **FLAG >0.02**), unmasked 0.0003 (delta 0.0000 OK) vs findings.md 0.0003
- (a) Parihaka client 3 (averaged BN): masked 0.1519 (delta 0.1515 **FLAG >0.02**), unmasked 0.0004 (delta 0.0000 OK) vs findings.md 0.0004
- (b) Parihaka client 2 (local BN): masked 0.7926 (delta 0.0319 **FLAG >0.02**), unmasked 0.7796 (delta 0.0189 OK) vs findings.md 0.7607
- (b) Parihaka client 3 (local BN): masked 0.6365 (delta 0.0380 **FLAG >0.02**), unmasked 0.6309 (delta 0.0324 **FLAG >0.02**) vs findings.md 0.5985

Interpretation of any FLAG: findings.md numbers are UNMASKED-argmax values (0.0003/0.0004 for (a) means the unmasked argmax never selects Parihaka rows under averaged BN); the masked (deployed) values score the same model within the Parihaka block only. For (b), a remaining unmasked delta is attributable to the BN refresh protocol (this run: one EMA-continuing pass over the train loader in train() mode, as specified in the figure brief).

- crossfade `f3_crossfade.mp4`/`.gif` (5.0 MB), 4.0 s @ 12 fps, (a) -> (b)

### F3 row (control)

- client 0, F3 crossline 313: (a) averaged BN masked mIoU 0.3238, (b) local BN masked mIoU 0.8546. On F3 the masked and unmasked conventions coincide (the model's argmax stays in rows 0-5). (a) is degraded vs (b) but structured - NOT the single-facies blackout seen on Parihaka; the weights are fine once BN is right. Findings.md also records the global model scoring only 0.10-0.19 on the F3 official test (broken-reference note).

## F4 - Training movie (20 rounds, facies appearing)

- run: `results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0` (config from local `run_meta.json`, seed 0, UNet(1,6,norm=batch,groups=8))
- round checkpoints: `/mnt/storage/results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0/received/round_001..020.pt` (all 20 present) — loaded as-is in eval mode = each round's own averaged BN running stats
- fixed crossline: client 2 site holdout, crossline 675 — median per-crossline present-class mIoU under round 20 over all 141 holdout crosslines (mIoU 0.9265; Q1 0.7683, Q3 0.9668; sample: c0 xl186: 0.977, c0 xl200: 0.958, c0 xl214: 0.980, c0 xl228: 0.975, c1 xl428: 0.926, c1 xl442: 0.708, c1 xl456: 0.710, c2 xl658: 0.771, c2 xl672: 0.917, c2 xl686: 0.938, c2 xl700: 0.943)
- movie: 20 frames (round 1..20) + 22 hold frames = 3.8 s @ 11 fps, H.264 -preset veryfast; GIF 1.0 MB (<=15 MB)
- artifacts: `f4_training_movie.mp4`/`.gif`/`_final.png`, `f4_rounds_grid.png`, round stills `f4_round01/05/10/20.png` (PNG 300dpi + PDF + SVG via style.save)
- device: cpu (GPU fully occupied by another job at run time)
