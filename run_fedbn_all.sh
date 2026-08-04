#!/bin/bash
# Run all FedBN experiments: Parihaka (6) + F3 (6) = 12 total
# Usage: bash run_fedbn_all.sh

set -e
cd /home/carlos/projects/FL-for-Seismic

echo "=== Starting FedBN experiments: $(date) ==="

echo "=========================================="
echo "  FedBN on Parihaka"
echo "=========================================="

echo "--- Parihaka FedBN 1/6: Non-IID 3 clients ---"
python train_fedbn.py --num_clients 3 --split noniid 2>&1

echo "--- Parihaka FedBN 2/6: IID 3 clients ---"
python train_fedbn.py --num_clients 3 --split iid 2>&1

echo "--- Parihaka FedBN 3/6: Non-IID 5 clients ---"
python train_fedbn.py --num_clients 5 --split noniid 2>&1

echo "--- Parihaka FedBN 4/6: IID 5 clients ---"
python train_fedbn.py --num_clients 5 --split iid 2>&1

echo "--- Parihaka FedBN 5/6: Non-IID 20 clients ---"
python train_fedbn.py --num_clients 20 --split noniid 2>&1

echo "--- Parihaka FedBN 6/6: IID 20 clients ---"
python train_fedbn.py --num_clients 20 --split iid 2>&1

echo "=========================================="
echo "  FedBN on F3"
echo "=========================================="

echo "--- F3 FedBN 1/6: Non-IID 3 clients ---"
python train_fedbn_f3.py --num_clients 3 --split noniid 2>&1

echo "--- F3 FedBN 2/6: IID 3 clients ---"
python train_fedbn_f3.py --num_clients 3 --split iid 2>&1

echo "--- F3 FedBN 3/6: Non-IID 5 clients ---"
python train_fedbn_f3.py --num_clients 5 --split noniid 2>&1

echo "--- F3 FedBN 4/6: IID 5 clients ---"
python train_fedbn_f3.py --num_clients 5 --split iid 2>&1

echo "--- F3 FedBN 5/6: Non-IID 20 clients ---"
python train_fedbn_f3.py --num_clients 20 --split noniid 2>&1

echo "--- F3 FedBN 6/6: IID 20 clients ---"
python train_fedbn_f3.py --num_clients 20 --split iid 2>&1

echo "=== All FedBN experiments complete: $(date) ==="
