# Distribution leakage in personalized cross-silo FL — status brief

Carlos Mari, Georgia Tech MS thesis (OLIVES Lab, advisor Prof. Ghassan AlRegib).
Working title: *Exploring the Boundaries of the Privacy–Personalization Tradeoff in Federated Learning for Seismic Data.*
Date: October 14, 2026. All claims below are measured; predictions were pre-registered in `findings.md` before each run.

## 1. Setting and priorities

Cross-silo FL: 2–10 organizations (e.g. companies holding seismic surveys) with strongly label-skewed data. Personalized FL exploits each client's label distribution π; that distribution is commercially sensitive.

Secret, in priority order: (1) **"how much"** — class counts n_k = n·π_k (counts, dataset size, proportions given support); (2) **support** — which classes are present (matters for rare classes); (3) location (seismic only, later).

Observers (always stated per result): **server-individual** (sees each update, no secure aggregation), **server-sum** (sees only the cohort sum), **peer-sum** (sees the sum, subtracts its own update). Honest-but-curious; malicious server out of scope. Protocol ceiling (externally reviewed): full stable participation with an encrypted sum; all schedule tricks are worse (SecAgg+/threshold-HE, signed round manifests).

## 2. Mechanism (first order)

A client's update is a **count-weighted sum of per-class gradient signatures**: Δ ≈ −ηΣ_k n_k ḡ_k. Four consequences: signatures (counts recoverable by unmixing), support (absent classes contribute ≈0 — no loss reweighting removes this), mixture (label-free statistics are π-weighted mixtures), sums (aggregation dilutes "how much" ~1/m, turns presence into the union).

## 3. Benchmarks

BloodMNIST, OrganCMNIST, CIFAR-10; small CNN **without BatchNorm**; Dirichlet α=0.1 (plus α=0.5 robustness); 5 seeds; SGD. Every number has 95% client-cluster bootstrap CIs (1000 draws), is compared to a prior-only floor, and uses **method-aware** shadows (victim's exact trainer, started from a received model saved during the run). FedAvg control must pass before any other method is interpreted.

## 4. FedLC (logit-adjusted training): two channels, opposite behavior

- **Bias channel dead.** Presence AUC ≈ 0.50 (chance) on all three datasets (FedAvg 0.82–0.87). Even told the true support, the bias cannot read fractions.
- **Row-size channel exposes presence.** AUC 0.98–1.00, TPR at 10% FPR 0.96–1.00 (floors 0.05–0.11). Absent-vs-≥0.25-share accuracy 0.89/0.93/0.96 (Blood/OrganC/CIFAR-10). A ~1% share is still detectable on OrganC/CIFAR-10.
- **Counts still leak through row sizes.** Counts R² 0.21 [0.12,0.26] / 0.27 [0.07,0.36] / 0.40 [0.33,0.41]. Dataset size does not (all R² ≤ 0), and standard FedAvg sends n_i as the aggregation weight anyway (protocol leak, not update leak).
- Prior smoothing π′=(1−λ)π+λ/K trades one leak for the other: best presence AUC never drops below 0.82 on Blood. No private operating point on this curve.
- A combined defense (inverse-frequency + 8 public images/class + local prior) fails: bias reverses (flipped AUC ≈0.90), accuracy 0.93→0.87. Not iterated.

## 5. Workstream B (closed): balanced gradients

One full-dataset step per round = mean of per-class mean gradients over classes present; log π only at inference. Counts/size/fractions sit **on the floor at 1, 2, and 5 steps on all three datasets** (row-size counts R² ≈ −0.04 everywhere; the "counts return with more steps" prediction failed). Support stays high on bias/Decaf (AUC 0.95–1.00) but falls to floor on row-size. Cost: balanced-5 local accuracy trails FedLC by 0.14/0.10/0.18.

## 6. Workstream C (closed on classification): few-silo sums, Blood

N ∈ {2,3,5,8} × α ∈ {0.1,0.5}, full participation, 20 seeds/cell, plaintext-simulated sums. Shadows are scale-matched (repeated Dirichlet(N) partitions) and regime-matched (rounds 1–10 victim means, shadows started in the same window — late-round updates come from a converged model and carry no slope).

- **Server-sum counts are on the floor at every N under both methods** (FedAvg −0.19→−0.05, FedLC −0.21→−0.05). Even N=2 kills "how much".
- **Server-sum support splits by method.** FedAvg sum presence ≈ floor (0.51–0.57). FedLC row-size presence stays 0.62–0.67 — the union-support effect (one present member lights the class for the whole sum). The encrypted sum hides how-much but **not** FedLC support; the method ranking reverses vs the individual observer.
- **Peer-sum counts**: N=2 equals individual by algebra; N=3 partial (Decaf +0.21); N≥5 dead. **Peer-sum support**: others'-union presence AUC 0.92–1.00 (all channels FedAvg; row/Decaf FedLC, bias at chance) for N≤5; N=8 saturates (union covers all classes).
- **Fingerprint** (10 clients, 3/round, hidden schedule): pairwise cohort-membership AUC — FedAvg row 0.85, FedLC row 0.99, decaf 0.81/0.96; bias weak. Peers recover their cohort under both methods.
- α=0.5 replicates where controls pass; N=2,3 cells gated (no within-seed count variance under mild skew).

## 7. Workstream G (in progress): seismic

F3 Netherlands, contiguous crossline sites (N=3,5). Site mixes vary realistically (c4: 0.01–0.07, c5 absent in some sites). UNet × (FedAvg / FedBN / GroupNorm). Timing run: global mIoU 0.55 vs deployed-local 0.98 — enormous personalization gap. BN running stats captured per upload (statistics channel); GroupNorm is the no-stats control. Found and fixed: head logging tracked the auxiliary `reconstruct` head instead of `outc`. Grid (24 runs) training now.

## 8. Open questions for an expert reader

1. The regime-matched attacker (rounds 1–10 both sides, justified by the label-free decay curve): is the window choice legitimate, or should shadows replicate full 40-round trajectory means?
2. Peer residual for counts at N≥5: is there a better-calibrated observer than the single-client line, or is "peers can't unmix ≥4 others" the substantive claim?
3. Given FedLC support survives the encrypted sum, is there any protocol-level mitigation short of hiding per-class update structure (e.g. neuronal shuffling), or is this inherent to logit-adjusted training?
4. Decentralized extension (all-to-all secure sum vs Swarm/ProxyFL/Shatter): does the support-survives-sum result change the ranking of these designs?

Code, configs, and per-run predictions are in the repo (`cursor/privacy-personalization-pipeline-a919`); `results/` tables are available on request (gitignored). Contact: Carlos Mari.
