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
python3 [path-to-repo-location]/main.py --seed=1 --partition_method="dirichlet" --partition_alpha=0.1 --n_rounds=200 --batch_size=50 --n_clients=100 --root='path-to-dataset' --dataset_name='cifar10' --model_name='fedavg_cifar' --base_folder='path-to-results-folder' --root_path='path-to-your-FL-repo' --sample_ratio=0.1 --date='enter-date-here' --config_path="/config/fedavg.json"
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

# Additional Resources
1. [Facies Classification Paper](https://arxiv.org/pdf/1901.07659)
