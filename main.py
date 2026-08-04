import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import algorithms
from train_tools import *
from utils import *
import numpy as np
import argparse
import warnings
import random
import pprint
import os
import wandb

warnings.filterwarnings("ignore")

# Set torch base print precision
torch.set_printoptions(10)

ALGO = {
    "fedavg": algorithms.fedavg.Server,
    "fedseismic": algorithms.fedseismic.Server
}

SCHEDULER = {
    "step": lr_scheduler.StepLR,
    "multistep": lr_scheduler.MultiStepLR,
    "cosine": lr_scheduler.CosineAnnealingLR,
}


def _get_setups(args):

    np.random.seed(args.train_setups.seed)
    random.seed(args.train_setups.seed)

    # Create save folder
    base_folder = args.data_setups.base_folder
    folder = str(args.data_setups.n_clients) + '_' + 'Clients_' + args.train_setups.algo.name + '_' + str(args.train_setups.scenario.sample_ratio)

    # Set up folder to save stuff
    seed = args.train_setups.seed
    if args.data_setups.dataset_name == 'seismic':
        # no dirichlet, etc partition in seismic semantic segmentation
        train_test_folder = (base_folder + '/' + args.data_setups.dataset_name + '/' +
                             str(args.data_setups.n_clients) + '_' + 'Clients_Idxs' + '/')
        save_folder = (base_folder + args.data_setups.date + '/' + args.data_setups.dataset_name + '/' +
                       folder + '/' + str(seed) + '/')
    else:
        if args.data_setups.partition.method == 'dirichlet':
            train_test_folder = (base_folder + '/' + args.data_setups.dataset_name + '/' + \
                          args.data_setups.partition.method + '_' + str(args.data_setups.partition.alpha) + '/' +
                                 str(args.data_setups.n_clients) + '_' + 'Clients_Idxs' + '/')

            save_folder = (base_folder + args.data_setups.date + '/' + args.data_setups.dataset_name + '/' +
                           args.data_setups.partition.method + '_' + str(args.data_setups.partition.alpha) + '/' + folder + '/' +
                           str(seed) + '/')

        elif args.data_setups.partition.method == 'sharding':
            train_test_folder = (base_folder + '/' + args.data_setups.dataset_name + '/' +
                                 args.data_setups.partition.method + '_' + str(args.data_setups.partition.shard_per_user) + '/' +
                                 str(args.data_setups.n_clients) + '_' + 'Clients_Idxs' + '/')
            save_folder = (base_folder + args.data_setups.date + '/' + args.data_setups.dataset_name + '/' +
                           args.data_setups.partition.method + '_' + str(args.data_setups.partition.shard_per_user) +
                           '/' + folder + '/' + str(seed) + '/')
        else:
            train_test_folder = base_folder + '/' + args.data_setups.dataset_name + '/' + \
                          args.data_setups.partition.method + '/' + str(args.data_setups.n_clients) + '_' + 'Clients_Idxs' + '/'
            save_folder = (base_folder + args.data_setups.date + '/' + args.data_setups.dataset_name + '/' +
                           args.data_setups.partition.method + '/' + folder + '/' + str(seed) + '/')

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if not os.path.exists(train_test_folder):
        os.makedirs(train_test_folder)

    # Distribute the data to clients
    # Returns dictionary with global and local loaders
    data_distributed = data_distributer(args=args, root=args.data_setups.root, dataset_name=args.data_setups.dataset_name,
                                        batch_size=args.data_setups.batch_size, n_clients=args.data_setups.n_clients,
                                        partition=args.data_setups.partition, save_folder=train_test_folder)

    _random_seeder(args.train_setups.seed)

    model = create_models(
        args.train_setups.model.name,
        args.data_setups.dataset_name
    )
    # Optimization setups
    optimizer = optim.SGD(model.parameters(), **args.train_setups.optimizer.params)
    scheduler = None

    if args.resume:
        checkpoint = torch.load(save_folder + 'model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if args.train_setups.scheduler.enabled:
        scheduler = SCHEDULER[args.train_setups.scheduler.name](
            optimizer, **args.train_setups.scheduler.params
        )

    dynamic_ratio = args.data_setups.local_setups["dynamic_ratio"]
    consistency = args.data_setups.local_setups["consistency"]
    dataset = args.data_setups.dataset_name

    algo_params = args.train_setups.algo.params

    if args.resume:
        print()
        print('>>> Resuming from Previous Checkpoint')
        stats = True
    else:
        stats = False

    #print('r: ', args.resume)

    server = ALGO[args.train_setups.algo.name](
        algo_params,
        model,
        data_distributed,
        optimizer,
        scheduler,
        dynamic_ratio,
        consistency,
        dataset,
        save_folder,
        stats,
        **args.train_setups.scenario,
    )

    return server, save_folder


def _random_seeder(seed):
    """Fix randomness"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """Execute experiment"""
    # Load the configuration
    server, path = _get_setups(args)

    # Conduct FL
    server.run()

    # Save the final global model
    model_path = os.path.join(path, "model.pth")
    torch.save(server.model.state_dict(), model_path)



######################################################################3

# Parser arguments for terminal execution
parser = argparse.ArgumentParser(description="Run FL Experiment from Config")
parser.add_argument("--config_path", required=True, type=str, help="Path to JSON config file")
parser.add_argument("--resume", action='store_true', help="Resume from checkpoint")
args = parser.parse_args()

if __name__ == "__main__":
    # Load configuration from .json file
    opt = ConfLoader(args.config_path).opt

    # Set resume flag if provided
    opt.resume = args.resume

    # Print configuration dictionary pretty
    print("")
    print("=" * 50 + " Configuration " + "=" * 50)
    pp = pprint.PrettyPrinter(compact=True)
    pp.pprint(opt)
    print("=" * 120)

    # Initialize wandb
    wandb.init(
        project=opt.wandb_setups.project,
        group=opt.wandb_setups.group,
        name=opt.wandb_setups.name,
        config=dict(opt)
    )

    # Execute experiment
    main(opt)

    # Finish wandb run
    wandb.finish()
