#!/bin/bash
# FedVLS on Non-IID Parihaka (3, 5, 20 clients)
set -e
cd /home/carlos/projects/FL-for-Seismic

echo "=== Starting FedVLS experiments: $(date) ==="

echo "--- FedVLS NonIID 3c lam=0.1 ---"
python train_fedvls.py --num_clients 3 --split noniid --lam 0.1 2>&1

echo "--- FedVLS NonIID 5c lam=0.1 ---"
python train_fedvls.py --num_clients 5 --split noniid --lam 0.1 2>&1

echo "--- FedVLS NonIID 20c lam=0.1 ---"
python train_fedvls.py --num_clients 20 --split noniid --lam 0.1 2>&1

echo "=== All FedVLS experiments complete: $(date) ==="
