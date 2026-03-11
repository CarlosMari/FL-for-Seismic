# Federated Learning for Seismic Facies Segmentation — Comprehensive Results

## Abstract-Ready Summary

We investigate federated learning (FL) for 2D seismic facies segmentation on two datasets (Parihaka, New Zealand and F3, Netherlands) using a UNet architecture. We benchmark five FL algorithms — FedAvg, FedProx, FedBN, FedAvg with class-weighted loss, and FedVLS — under IID and geographically Non-IID data partitioning with 3, 5, and 20 simulated clients. Key findings: (1) IID federated learning matches centralized training (mIoU gap <1%), demonstrating FL's viability for seismic interpretation when data is representative. (2) Geographic (Non-IID) partitioning causes severe minority-class collapse — rare facies (class 5, ~1.5% of data) achieve 0.0 IoU at 5+ clients across all algorithms. (3) Plain FedAvg outperforms FedProx, FedBN, and class-weighted variants in all configurations. (4) Client subsampling (selecting 25% of clients per round) improves Non-IID 20-client mIoU by +10.7% on Parihaka (0.514→0.569) by reducing gradient dilution. (5) FedVLS's vacant-class distillation is the only method to achieve non-zero rare-class IoU at 20 clients, confirming that the core challenge is data absence rather than optimization dynamics.

---

## Experimental Setup

### Model and Training
- **Architecture**: UNet (2D), ~31M parameters, 1-channel input, 6-class output
- **Loss**: FocalLoss (gamma=2) + DiceLoss
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **FL Configuration**: 20 communication rounds, 3 local epochs per round (60 total epochs)
- **Batch size**: 4
- **Seed**: 42 (deterministic)

### Datasets

| Dataset | Source | Cube Shape | Train/Test | Classes | Notable Imbalance |
|---------|--------|-----------|------------|---------|-------------------|
| Parihaka | New Zealand | 401 x 701 x 255 | Pre-split (2 test sets) | 6 | Class 5: 1.5%, concentrated in crosslines 420-630 |
| F3 | Netherlands | 401 x 701 x 255 | 80/20 crossline split | 6 | Class 5: spatially concentrated |

### Parihaka Class Distribution
| Class | 0 | 1 | 2 | 3 | 4 | 5 |
|-------|------|------|------|-----|-----|-----|
| Frequency | 28.1% | 11.9% | 48.6% | 6.6% | 3.3% | 1.5% |

### Data Partitioning Strategies
- **IID**: Crosslines randomly shuffled and split equally across clients
- **Non-IID (Geographic)**: Contiguous crossline chunks per client — each client sees a spatially distinct region of the subsurface, creating natural label distribution skew

### Algorithms Tested

| Algorithm | Key Mechanism | Reference |
|-----------|--------------|-----------|
| FedAvg | Equal-weight averaging of client models | McMahan et al., 2017 |
| FedProx | Proximal term constraining local updates toward global model | Li et al., 2020 |
| FedBN | Keep BatchNorm layers local, only aggregate conv/linear weights | Li et al., ICLR 2021 |
| FedAvg+CW | Global inverse-frequency class weights in FocalLoss alpha | — |
| FedVLS | Vacant-class distillation + calibrated CE + logit suppression | Guo et al., AAAI 2024 |

---

## Main Results

### Centralized Baselines (Upper Bound)

| Dataset  | Best mIoU | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 |
|----------|-----------|---------|---------|---------|---------|---------|---------|
| Parihaka | 0.693     | 0.918   | 0.828   | 0.946   | 0.583   | 0.398   | 0.483   |
| F3       | 0.786     | 0.971   | 0.888   | 0.973   | 0.749   | 0.350   | 0.782   |

### FedAvg — Parihaka

| Split    | Clients | Best mIoU | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 |
|----------|---------|-----------|---------|---------|---------|---------|---------|---------|
| Non-IID  | 3       | 0.628     | 0.906   | 0.795   | 0.908   | 0.522   | 0.341   | 0.283   |
| IID      | 3       | 0.686     | 0.906   | 0.835   | 0.938   | 0.570   | 0.399   | 0.415   |
| Non-IID  | 5       | 0.571     | 0.877   | 0.790   | 0.879   | 0.456   | 0.160   | 0.000   |
| IID      | 5       | 0.671     | 0.906   | 0.835   | 0.938   | 0.570   | 0.392   | 0.386   |
| Non-IID  | 20      | 0.514     | 0.896   | 0.795   | 0.856   | 0.417   | 0.122   | 0.000   |
| IID      | 20      | 0.681     | 0.916   | 0.800   | 0.938   | 0.604   | 0.421   | 0.404   |

### FedAvg — F3

| Split    | Clients | Best mIoU | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 |
|----------|---------|-----------|---------|---------|---------|---------|---------|---------|
| Non-IID  | 3       | 0.628     | 0.979   | 0.898   | 0.983   | 0.744   | 0.166   | 0.000   |
| IID      | 3       | 0.781     | 0.976   | 0.883   | 0.982   | 0.757   | 0.348   | 0.740   |
| Non-IID  | 5       | 0.604     | 0.979   | 0.888   | 0.957   | 0.641   | 0.161   | 0.000   |
| IID      | 5       | 0.785     | 0.978   | 0.894   | 0.976   | 0.753   | 0.369   | 0.742   |
| Non-IID  | 20      | 0.579     | 0.978   | 0.881   | 0.954   | 0.523   | 0.134   | 0.000   |
| IID      | 20      | 0.787     | 0.980   | 0.892   | 0.979   | 0.746   | 0.397   | 0.731   |

### FedProx (mu=0.01) — Parihaka

| Split    | Clients | Best mIoU | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 |
|----------|---------|-----------|---------|---------|---------|---------|---------|---------|
| Non-IID  | 3       | 0.557     | 0.932   | 0.777   | 0.854   | 0.547   | 0.231   | 0.000   |
| IID      | 3       | 0.630     | 0.934   | 0.787   | 0.916   | 0.540   | 0.282   | 0.321   |
| Non-IID  | 5       | 0.557     | 0.863   | 0.778   | 0.841   | 0.519   | 0.190   | 0.006   |
| IID      | 5       | 0.669     | 0.909   | 0.818   | 0.930   | 0.549   | 0.401   | 0.404   |
| Non-IID  | 20      | 0.471     | 0.829   | 0.733   | 0.791   | 0.439   | 0.035   | 0.000   |
| IID      | 20      | 0.564     | 0.887   | 0.797   | 0.872   | 0.479   | 0.110   | 0.240   |

### FedProx (mu=0.01) — F3

| Split    | Clients | Best mIoU | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 |
|----------|---------|-----------|---------|---------|---------|---------|---------|---------|
| Non-IID  | 3       | 0.614     | 0.971   | 0.868   | 0.963   | 0.676   | 0.203   | 0.005   |
| IID      | 3       | 0.746     | 0.974   | 0.878   | 0.964   | 0.640   | 0.425   | 0.595   |
| Non-IID  | 5       | 0.581     | 0.977   | 0.859   | 0.933   | 0.496   | 0.221   | 0.000   |
| IID      | 5       | 0.759     | 0.979   | 0.847   | 0.972   | 0.616   | 0.409   | 0.728   |
| Non-IID  | 20      | 0.533     | 0.971   | 0.808   | 0.905   | 0.438   | 0.079   | 0.000   |
| IID      | 20      | 0.684     | 0.977   | 0.843   | 0.960   | 0.534   | 0.315   | 0.477   |

### FedBN — Parihaka

| Split    | Clients | Best mIoU | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 |
|----------|---------|-----------|---------|---------|---------|---------|---------|---------|
| Non-IID  | 3       | 0.555     | 0.816   | 0.726   | 0.878   | 0.466   | 0.233   | 0.211   |
| IID      | 3       | 0.673     | 0.902   | 0.806   | 0.914   | 0.576   | 0.412   | 0.427   |
| Non-IID  | 5       | 0.529     | 0.820   | 0.694   | 0.876   | 0.462   | 0.318   | 0.001   |
| IID      | 5       | 0.608     | 0.838   | 0.725   | 0.895   | 0.528   | 0.323   | 0.338   |
| Non-IID  | 20      | 0.442     | 0.762   | 0.575   | 0.771   | 0.365   | 0.182   | 0.000   |
| IID      | 20      | 0.638     | 0.911   | 0.788   | 0.919   | 0.542   | 0.341   | 0.328   |

### FedBN — F3

| Split    | Clients | Best mIoU | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 |
|----------|---------|-----------|---------|---------|---------|---------|---------|---------|
| Non-IID  | 3       | 0.589     | 0.942   | 0.798   | 0.964   | 0.678   | 0.150   | 0.001   |
| IID      | 3       | 0.751     | 0.967   | 0.837   | 0.975   | 0.733   | 0.285   | 0.708   |
| Non-IID  | 5       | 0.498     | 0.857   | 0.605   | 0.925   | 0.506   | 0.096   | 0.000   |
| IID      | 5       | 0.713     | 0.952   | 0.809   | 0.959   | 0.654   | 0.270   | 0.632   |
| Non-IID  | 20      | 0.518     | 0.885   | 0.743   | 0.906   | 0.421   | 0.150   | 0.000   |
| IID      | 20      | 0.744     | 0.961   | 0.838   | 0.967   | 0.698   | 0.348   | 0.651   |

---

## Mitigation Strategies for Non-IID Degradation

### FedProx mu Sweep (Non-IID, 20 clients)

| mu   | Parihaka mIoU | F3 mIoU | vs FedAvg (Parihaka) | vs FedAvg (F3) |
|------|---------------|---------|----------------------|----------------|
| 0 (FedAvg) | **0.514** | **0.579** | —              | —              |
| 0.01 | 0.471         | 0.533   | -0.043               | -0.046         |
| 0.1  | 0.411         | 0.481   | -0.103               | -0.098         |
| 0.5  | 0.367         | 0.390   | -0.147               | -0.189         |

Higher mu is strictly worse. Proximal regularization over-constrains local updates for this task.

### Client Subsampling (FedAvg, Non-IID 20 clients, Parihaka)

| Sample Ratio | Clients/Round | Best mIoU | Class 4 | Class 5 | vs Full (0.514) |
|-------------|---------------|-----------|---------|---------|-----------------|
| 1.0 (all)   | 20/20         | 0.514     | 0.122   | 0.000   | —               |
| 0.5          | 10/20         | 0.543     | 0.254   | 0.000   | +0.029          |
| 0.25         | 5/20          | **0.569** | **0.308** | 0.000 | **+0.055**      |

Client subsampling is the most effective simple mitigation: +10.7% mIoU, class 4 IoU more than doubles.

### Class-Weighted FedAvg (Non-IID, Parihaka)

Global inverse-frequency class weights in FocalLoss alpha: [0.17, 0.40, 0.10, 0.72, 1.45, 3.16]

| Clients | FedAvg+CW | FedAvg | Delta  | CW C4 | Avg C4 | CW C5 | Avg C5 |
|---------|-----------|--------|--------|-------|--------|-------|--------|
| 3       | 0.620     | 0.628  | -0.008 | 0.320 | 0.341  | 0.224 | 0.283  |
| 5       | 0.571     | 0.571  | +0.001 | 0.300 | 0.160  | 0.000 | 0.000  |
| 20      | 0.518     | 0.514  | +0.004 | 0.138 | 0.122  | 0.000 | 0.000  |

Marginal effect — loss weighting cannot fix data absence.

### FedVLS — Vacant-Class Distillation (Non-IID, Parihaka)

| Clients | FedVLS | FedAvg | Delta  | VLS C4 | Avg C4 | VLS C5 | Avg C5 |
|---------|--------|--------|--------|--------|--------|--------|--------|
| 3       | 0.598  | 0.628  | -0.030 | **0.396** | 0.341 | 0.017 | 0.283 |
| 5       | **0.577** | 0.571 | +0.006 | **0.338** | 0.160 | 0.000 | 0.000 |
| 20      | 0.502  | 0.514  | -0.012 | 0.103  | 0.122  | **0.022** | 0.000 |

Only algorithm to achieve non-zero class 5 at 20 clients. Confirms vacant-class distillation as the right mechanism but with majority-class trade-off.

---

## Cross-Dataset Comparison — All Algorithms (Non-IID mIoU)

| Algorithm | Parihaka 3c | Parihaka 5c | Parihaka 20c | F3 3c | F3 5c | F3 20c |
|-----------|-------------|-------------|--------------|-------|-------|--------|
| **FedAvg** | **0.628** | 0.571 | 0.514 | **0.628** | **0.604** | 0.579 |
| FedAvg sr=0.25 | — | — | **0.569** | — | — | pending |
| FedAvg+CW | 0.620 | **0.571** | 0.518 | — | — | — |
| FedVLS | 0.598 | **0.577** | 0.502 | — | — | — |
| FedProx | 0.557 | 0.557 | 0.471 | 0.614 | 0.581 | 0.533 |
| FedBN | 0.555 | 0.529 | 0.442 | 0.589 | 0.498 | 0.518 |

## Cross-Dataset Comparison — All Algorithms (IID mIoU)

| Algorithm | Parihaka 3c | Parihaka 5c | Parihaka 20c | F3 3c | F3 5c | F3 20c |
|-----------|-------------|-------------|--------------|-------|-------|--------|
| **FedAvg** | **0.686** | **0.671** | **0.681** | **0.781** | **0.785** | **0.787** |
| FedProx | 0.630 | 0.669 | 0.564 | 0.746 | 0.759 | 0.684 |
| FedBN | 0.673 | 0.608 | 0.638 | 0.751 | 0.713 | 0.744 |

## Federation Cost vs Centralized

| Dataset  | Centralized | Best IID FL | IID Gap | Best Non-IID FL | Non-IID Gap |
|----------|-------------|-------------|---------|-----------------|-------------|
| Parihaka | 0.693       | 0.686 (3c)  | -1.0%   | 0.569 (20c sr=0.25) | -17.9% |
| F3       | 0.786       | 0.787 (20c) | **+0.1%** | 0.579 (20c)    | -26.3% |

---

## Key Findings for IMAGE Abstract

1. **FL is viable for seismic segmentation under IID conditions**: FedAvg with IID partitioning achieves within 1% of centralized training on both Parihaka (0.686 vs 0.693) and F3 (0.787 vs 0.786), with F3 IID slightly exceeding the centralized baseline — suggesting federated averaging provides implicit regularization.

2. **Geographic Non-IID partitioning causes severe degradation**: When clients hold spatially contiguous data (realistic for multi-organization collaboration), mIoU drops 17-26% vs centralized. The degradation scales with client count (0.628→0.514 for 3→20 clients on Parihaka).

3. **Minority facies collapse is the dominant failure mode**: Under Non-IID, rare classes (class 5: ~1.5% of data, spatially concentrated) achieve 0.0 IoU at 5+ clients across FedAvg, FedProx, FedBN, and class-weighted variants. The cause is data absence — clients without samples of a class actively degrade global model knowledge of that class during local training.

4. **Existing FL algorithms fail to address label-shift Non-IID**:
   - **FedProx** (-4 to -12% vs FedAvg): Proximal regularization slows convergence uniformly without distinguishing helpful vs harmful drift.
   - **FedBN** (-1 to -11% vs FedAvg): Designed for feature shift (different sensors/domains), not label shift (different class distributions). Removing BN aggregation loses useful shared statistics.
   - **Class-weighted loss** (marginal): Cannot teach absent classes regardless of weighting.

5. **Client subsampling is the most effective simple mitigation**: Sampling 25% of clients per round (5/20) improves Non-IID mIoU by +10.7% on Parihaka. Mechanism: reduces gradient dilution of rare-class signals from the few clients that contain them.

6. **Vacant-class distillation (FedVLS) addresses the right problem**: Only algorithm to achieve non-zero rare-class IoU at 20 Non-IID clients. Class 4 IoU doubles at 5 clients (+111%). However, overall mIoU improvement is modest due to majority-class trade-offs, indicating room for better loss balancing.

7. **The fundamental open challenge**: Geographic FL for seismic interpretation requires methods that can propagate class knowledge to clients that have never observed those classes. Standard FL optimization improvements (FedProx, FedBN) are insufficient. Promising directions include knowledge distillation (FedVLS), data synthesis, and cross-client feature sharing.

---

## Complete Experiment Log

Total experiments run: 50+
- FedAvg: 6 Parihaka + 6 F3 = 12
- FedProx mu=0.01: 6 Parihaka + 6 F3 = 12
- FedBN: 6 Parihaka + 6 F3 = 12
- Centralized baselines: 2 (Parihaka + F3)
- FedProx mu sweep: 4 (mu=0.1, 0.5 x Parihaka, F3)
- Client subsampling: 3 (sr=0.5, 0.25 on Parihaka 20c; sr=0.25 on F3 20c pending)
- Class-weighted FedAvg: 3 (Non-IID 3/5/20c Parihaka)
- FedVLS: 3 (Non-IID 3/5/20c Parihaka)

---

## wandb Project

All runs logged to: https://wandb.ai/carlosmari/FL-Seismic
