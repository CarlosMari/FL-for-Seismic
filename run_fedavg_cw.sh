#!/bin/bash
# FedAvg with class-weighted loss on Non-IID Parihaka (3, 5, 20 clients)
set -e
cd /home/carlos/projects/FL-for-Seismic

echo "=== Starting FedAvg class-weighted experiments: $(date) ==="

echo "--- FedAvg CW NonIID 3c ---"
python train_federated.py --num_clients 3 --split noniid --class_weights 2>&1

echo "--- FedAvg CW NonIID 5c ---"
python train_federated.py --num_clients 5 --split noniid --class_weights 2>&1

echo "--- FedAvg CW NonIID 20c ---"
python train_federated.py --num_clients 20 --split noniid --class_weights 2>&1

echo "=== All FedAvg CW experiments complete: $(date) ==="
