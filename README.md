# Overview
This GitHub project is meant to serve as a basic repository for federated learning on natural images and seismic data.

# Install Environment
You will first need to create a conda environment to run experiments. You can install the basic libraries needed as follows:

```
conda create --name <env_name> --file requirements.txt
```

If another library is needed that is not yet installed, you can do so by using ```pip```.

# Example Setups
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

# Additional Resources
1. [Facies Classification Paper](https://arxiv.org/pdf/1901.07659)
