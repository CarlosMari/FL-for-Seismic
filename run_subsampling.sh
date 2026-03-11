#!/bin/bash
# Client subsampling (sr=0.25) on all Non-IID configurations
# Parihaka 3c, 5c already missing; F3 3c, 5c, 20c all missing
set -e
cd /home/carlos/projects/FL-for-Seismic

echo "=== Starting subsampling experiments: $(date) ==="

echo "--- Parihaka NonIID 3c sr=0.25 ---"
python train_federated.py --num_clients 3 --split noniid --sample_ratio 0.25 2>&1

echo "--- Parihaka NonIID 5c sr=0.25 ---"
python train_federated.py --num_clients 5 --split noniid --sample_ratio 0.25 2>&1

echo "--- F3 NonIID 3c sr=0.25 ---"
python train_federated_f3.py --num_clients 3 --split noniid --sample_ratio 0.25 2>&1

echo "--- F3 NonIID 5c sr=0.25 ---"
python train_federated_f3.py --num_clients 5 --split noniid --sample_ratio 0.25 2>&1

echo "--- F3 NonIID 20c sr=0.25 ---"
python train_federated_f3.py --num_clients 20 --split noniid --sample_ratio 0.25 2>&1

echo "=== All subsampling experiments complete: $(date) ==="
