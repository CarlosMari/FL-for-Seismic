# Federated Learning for Seismic Facies Segmentation: Benchmarking Algorithms Under Geographic Data Heterogeneity

**Carlos Mari**, [co-authors and affiliations]

## Summary

Federated learning (FL) enables collaborative model training across organizations without sharing raw seismic data, addressing data privacy and ownership concerns in subsurface characterization. However, real-world FL deployments face geographic data heterogeneity: each participant holds seismic data from a spatially distinct region, creating non-identical label distributions across clients. We present a systematic benchmark of five FL algorithms for 2D seismic facies segmentation, evaluating their robustness to this geographic non-IID challenge.

We train a UNet architecture on two public datasets — Parihaka (New Zealand) and F3 (Netherlands) — under IID and geographic (non-IID) partitioning with 3, 5, and 20 simulated clients. We compare FedAvg, FedProx, FedBN, class-weighted FedAvg, and FedVLS across over 50 experiments, measuring six-class mean intersection-over-union (mIoU).

Our results reveal that IID federated learning matches centralized training within 1% on both datasets (0.686 vs 0.693 on Parihaka; 0.787 vs 0.786 on F3), confirming FL's viability when data is representative. However, geographic partitioning causes severe degradation: mIoU drops 18–26% versus centralized, with rare facies classes collapsing to 0.0 IoU when spatially concentrated in few clients. Critically, FedAvg outperforms all tested alternatives — FedProx and FedBN degrade performance by 1–12%, and class-weighted loss has negligible effect. We identify client subsampling (selecting 25% of clients per round) as the most effective simple mitigation, improving non-IID mIoU by up to 10.7% by reducing gradient dilution. FedVLS, which distills global model knowledge about absent classes during local training, is the only method to partially recover rare-class performance at high client counts.

We conclude that the dominant failure mode in geographic FL for seismic interpretation is data absence — clients lacking certain facies classes actively degrade global knowledge — rather than optimization dynamics. Future work should explore knowledge distillation and cross-client feature sharing to address this fundamental challenge.
