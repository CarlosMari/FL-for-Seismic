#!/bin/bash
# Run all F3 experiments: FedAvg (6) + FedProx (6) = 12 total
# Usage: bash run_f3_all.sh

set -e
cd /home/carlos/projects/FL-for-Seismic

echo "=== Starting F3 experiments: $(date) ==="

echo "=========================================="
echo "  FedAvg on F3"
echo "=========================================="

echo "--- F3 FedAvg 1/6: Non-IID 3 clients ---"
python train_federated_f3.py --num_clients 3 --split noniid 2>&1

echo "--- F3 FedAvg 2/6: IID 3 clients ---"
python train_federated_f3.py --num_clients 3 --split iid 2>&1

echo "--- F3 FedAvg 3/6: Non-IID 5 clients ---"
python train_federated_f3.py --num_clients 5 --split noniid 2>&1

echo "--- F3 FedAvg 4/6: IID 5 clients ---"
python train_federated_f3.py --num_clients 5 --split iid 2>&1

echo "--- F3 FedAvg 5/6: Non-IID 20 clients ---"
python train_federated_f3.py --num_clients 20 --split noniid 2>&1

echo "--- F3 FedAvg 6/6: IID 20 clients ---"
python train_federated_f3.py --num_clients 20 --split iid 2>&1

echo "=========================================="
echo "  FedProx on F3"
echo "=========================================="

echo "--- F3 FedProx 1/6: Non-IID 3 clients ---"
python train_fedprox_f3.py --num_clients 3 --split noniid --mu 0.01 2>&1

echo "--- F3 FedProx 2/6: IID 3 clients ---"
python train_fedprox_f3.py --num_clients 3 --split iid --mu 0.01 2>&1

echo "--- F3 FedProx 3/6: Non-IID 5 clients ---"
python train_fedprox_f3.py --num_clients 5 --split noniid --mu 0.01 2>&1

echo "--- F3 FedProx 4/6: IID 5 clients ---"
python train_fedprox_f3.py --num_clients 5 --split iid --mu 0.01 2>&1

echo "--- F3 FedProx 5/6: Non-IID 20 clients ---"
python train_fedprox_f3.py --num_clients 20 --split noniid --mu 0.01 2>&1

echo "--- F3 FedProx 6/6: IID 20 clients ---"
python train_fedprox_f3.py --num_clients 20 --split iid --mu 0.01 2>&1

echo "=== All F3 experiments complete: $(date) ==="
