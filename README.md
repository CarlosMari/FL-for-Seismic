# FL-for-Seismic: privacy–personalization boundary in federated learning

> **Status:** basis of my MS thesis and a journal paper in preparation (both expected December 2026).

This repo studies what a client's model updates reveal about its data
composition in cross-silo federated learning: class counts ("how much"),
which classes are present (support), and how personalization methods
(FedAvg, FedPer, LA [logit-adjusted, mislabelled FedLC], FedBABU, FedRoD, FedKPer, Ditto, balanced gradients)
change that leakage — on BloodMNIST, OrganCMNIST, CIFAR-10, and the F3
Netherlands seismic survey. `findings.md` is the lab notebook (dated
prediction-vs-outcome entries); `metrics.md` defines every score;
`todo.md` tracks the work plan.

# Install

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Versions in `requirements.txt` are pinned to the ones the reported results
were produced with (torch 2.9, scikit-learn 1.8, pandas 3.0, …).
A CUDA GPU is used when available (`--device cuda`).

# Layout

- `fedseismic/` — the current package: `config.py` (`RunConfig`),
  `experiment.py` (training entry point), `federated/` (server + client
  trainers), `models/` (CNNs, UNet), `data/` (MedMNIST/CIFAR/seismic
  loaders + partitions), `privacy/` (honest-but-curious attackers),
  `eval/` (metrics).
- `configs/` — single-run configs; `configs/sweeps/` — grid specs.
- `tests/` — 77 unit tests (`python -m unittest discover -s tests`).
- `results/` — run artifacts (gitignored, regenerated locally).
- `main.py`, `algorithms/`, `train_tools/` — the earlier training
  framework (see "Legacy framework" below).

# Reproducing the privacy grids

Single run, then the attack suite on it:

```
python -m fedseismic.cli run --config configs/fedkper_bloodmnist.json --output-dir results/demo --device cuda
python -m fedseismic.cli attack --run-dir results/demo
```

Full grids (training + attacks + tables + figures):

```
python -m fedseismic.cli sweep --config configs/sweeps/balanced_blood.json --device cuda --skip-attack
python -m fedseismic.privacy.howmuch --extra results/runs/balanced_blood --output results/tables/balanced_howmuch.csv
python -m fedseismic.privacy.sum_channel --roots results/runs/sums_blood results/runs/sums_blood_fedlc --output results/tables/sums_blood_a01.csv --alphas 0.1
python -m fedseismic.privacy.sum_channel --roots results/runs/sums_blood --output results/tables/sums_blood_a01.csv --figure
python -m fedseismic.privacy.fingerprint --roots results/runs/fingerprint_blood results/runs/fingerprint_blood_fedlc --output results/tables/fingerprint_blood.csv
```

Key grids: `balanced_{blood,organc,cifar}.json` (balanced gradients),
`sums_blood{,_fedlc}.json` (few-silo encrypted sums, N=2/3/5/8),
`fingerprint_blood{,_fedlc}.json` (cohort recovery, 10 clients 3/round),
`prior_smooth_{blood,organc,cifar}.json` (LA→FedAvg λ curve; LA = logit adjustment, not FedLC),
`seismic_f3.json` (F3 UNet, FedAvg/FedBN × batch/group norm).
Observers are always explicit: server-individual, server-sum, peer-sum.
Method-aware shadows (victim's exact trainer from a saved received model)
are cached under `results/attacks/shadows/`.

# Overview (legacy framework)

# Install Environment (legacy framework)
You will first need to create a conda environment to run experiments. You can install the basic libraries needed as follows:

```
conda create --name <env_name> --file requirements.txt
```

If another library is needed that is not yet installed, you can do so by using ```pip```.

# Example Setups (legacy framework)
Federated learning is an iterative process. It takes place across multiple communication rounds. At each round, we sample a certain percentage of all clients.
We are easily able to set these parameters using this codebase.
For instance, say we want to run the ```FedAvg``` algorithm. We can set up a bash script to contain this line:

```
python3 [path-to-repo-location]/main.py --seed=1 --partition_method="dirichlet" --partition_alpha=0.1 --n_rounds=10 --batch_size=50 --n_clients=100 --root='path-to-dataset' --dataset_name='seismic' --model_name='fedseismic' --base_folder='.results/' --root_path='./' --sample_ratio=0.1 --date='02-01-26' --config_path="/config/fedseismic.json"
```

In the above example, ```--sample_ratio``` is the percentage of clients sampled each round. ```--n_rounds``` is the number of total communication rounds.
```--n_clients``` is the total number of clients created. 
You can control the exact algorithm you are running by changing the ```--config_path```.

In federated learning, we also simulate label heterogeneity experiments, where we
purposefully make the clients have different label distributions.
For instance, maybe client 0 has classes 0 and 1, while client 1 has classes 2 and 3.
Clients having heterogeneous label distributions tends to cause the performance of FL algorithms to deteriorate.
One of the ways we simulate this data heterogeneity is via a Dirichlet distribution (```--partition_method='dirichlet'```), which is controlled
by an alpha parameter (```--partition_alpha```) that makes the client partition more
heterogeneous.

# Privacy–personalization boundary
The `fedseismic` entry point can log the honest-but-curious server view (weight
deltas, last-layer tensors, BatchNorm stats if present) separately from each
client's true label histogram π.

FedKPer aggregation no longer assumes the server already knows π. Set
`fedkper_diversity` to:

- `infer` (default): estimate π from the uploaded model on a public unlabeled probe set
- `upload`: the client sends its histogram with the update (direct leak)
- `oracle`: simulator upper bound using partition labels (not a real protocol)

BloodMNIST first, then the same attacker on seismic pixel histograms (including
rare-class 4/5 presence):

```
python -m fedseismic.cli run --config configs/fedkper_bloodmnist.json --output-dir results/demo --device cpu
python -m fedseismic.cli attack --run-dir results/demo
python -m fedseismic.cli sweep --config configs/sweeps/privacy_boundary.json --device cpu
python -m fedseismic.cli plot --csv results/privacy_boundary/summary.csv --out results/privacy_boundary/curve.png
```

Seismic uses `configs/sweeps/privacy_boundary_seismic.json`. The plotted curve is
personalization gap vs leakage `L = 1 - TV(π, π̂)`.

See [metrics.md](metrics.md) for how each score is calculated and what it means.

# Additional Resources
1. [Facies Classification Paper](https://arxiv.org/pdf/1901.07659)
