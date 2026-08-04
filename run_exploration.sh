#!/bin/bash
# Exploration experiments:
# 1. Centralized baselines (Parihaka + F3)
# 2. FedProx mu sweep (0.1, 0.5) on Non-IID 20c (Parihaka + F3)
# 3. FedAvg client subsampling (sample_ratio=0.5) on Non-IID 20c (Parihaka)

set -e
cd /home/carlos/projects/FL-for-Seismic

echo "=== Starting exploration experiments: $(date) ==="

echo "=========================================="
echo "  STEP 1: Centralized Baselines"
echo "=========================================="

echo "--- Centralized Parihaka (60 epochs) ---"
python train_centralized.py --dataset parihaka --epochs 60 2>&1

echo "--- Centralized F3 (60 epochs) ---"
python train_centralized.py --dataset f3 --epochs 60 2>&1

echo "=========================================="
echo "  STEP 2: FedProx mu sweep (Non-IID 20c)"
echo "=========================================="

echo "--- FedProx mu=0.1 NonIID 20c Parihaka ---"
python train_fedprox.py --num_clients 20 --split noniid --mu 0.1 2>&1

echo "--- FedProx mu=0.5 NonIID 20c Parihaka ---"
python train_fedprox.py --num_clients 20 --split noniid --mu 0.5 2>&1

echo "--- FedProx mu=0.1 NonIID 20c F3 ---"
python train_fedprox_f3.py --num_clients 20 --split noniid --mu 0.1 2>&1

echo "--- FedProx mu=0.5 NonIID 20c F3 ---"
python train_fedprox_f3.py --num_clients 20 --split noniid --mu 0.5 2>&1

echo "=========================================="
echo "  STEP 3: Client Subsampling (Non-IID 20c)"
echo "=========================================="

echo "--- FedAvg NonIID 20c sample_ratio=0.5 Parihaka ---"
python train_federated.py --num_clients 20 --split noniid --sample_ratio 0.5 2>&1

echo "--- FedAvg NonIID 20c sample_ratio=0.25 Parihaka ---"
python train_federated.py --num_clients 20 --split noniid --sample_ratio 0.25 2>&1

echo "=== All exploration experiments complete: $(date) ==="
