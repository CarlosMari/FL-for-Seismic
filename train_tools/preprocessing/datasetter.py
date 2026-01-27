import glob
import scipy
import torch
from collections import Counter
import random
import numpy as np
import os
from scipy.stats import mode
from .cifar10.loader import get_all_targets_cifar10, get_dataloader_cifar10
from .seismic.loader import get_seismic_idxs, get_dataloader_seismic

__all__ = ["data_distributer"]

DATA_INSTANCES = {
    "cifar10": get_all_targets_cifar10,
}

DATA_LOADERS = {
    "cifar10": get_dataloader_cifar10,
    "seismic": get_dataloader_seismic
}


def data_distributer(
    args,
    root,
    dataset_name,
    batch_size,
    n_clients,
    partition,
    save_folder
):
    """
    Distribute dataloaders for server and locals by the given partition method.
    """
    root = os.path.join(root, dataset_name)
    # Get all available classes for train samples
    if dataset_name != 'seismic':
        all_targets = DATA_INSTANCES[dataset_name](root, dataset_label=dataset_name)
    # Figure out number of classes
    if dataset_name == 'olives':
        num_classes = all_targets.shape[1]
    elif dataset_name == 'seismic':
        num_classes = 6
    else:
        num_classes = len(np.unique(all_targets))

    print('Class count: ', num_classes)

    net_dataidx_map_test = None

    local_loaders = {
        i: {"datasize": 0, "train": None, "test": None, "test_size": 0, "class_distribution": {}} for i in range(n_clients)
    }

    contents = glob.glob(save_folder + '*')

    if dataset_name != 'seismic':
        if partition.method == "centralized":
            net_dataidx_map = centralized_partition(all_targets)
        elif partition.method == "iid":
            net_dataidx_map = iid_partition(all_targets, n_clients)
            net_dataidx_map_test, net_dataidx_map = create_local(idxs=net_dataidx_map, all_targets=all_targets,
                                                                 save_folder=save_folder)

        elif partition.method == "lda":
            net_dataidx_map = lda_partition(all_targets, n_clients, partition.alpha)
            net_dataidx_map_test, net_dataidx_map = create_local(idxs=net_dataidx_map, all_targets=all_targets,
                                                                 save_folder=save_folder)
        elif partition.method == 'dirichlet':
            net_dataidx_map = partition_class_samples_with_dirichlet_distribution(alpha=partition.alpha, client_num=n_clients,
                                                                                  targets=all_targets, class_num=num_classes)
            net_dataidx_map_test, net_dataidx_map = create_local(idxs=net_dataidx_map, all_targets=all_targets,
                                                                 save_folder=save_folder)
        else:
            raise NotImplementedError
    else:
        # For Seismic: segmentation task so dirichlet etc doesn't make sense
        net_dataidx_map, net_dataidx_map_test = get_seismic_idxs(root, n_clients)
        # Save idxs
        np.save(save_folder + 'train_idxs.npy', net_dataidx_map, allow_pickle=True)
        np.save(save_folder + 'test_idxs.npy', net_dataidx_map_test, allow_pickle=True)

    print(">>> Distributing client train data...")
    print(save_folder)
    for client_idx, dataidxs in net_dataidx_map.items():
        if dataset_name != 'seismic':
            local_train = DATA_LOADERS[dataset_name](
                root, mode='tr', batch_size=batch_size, dataidxs=dataidxs, dataset_label=dataset_name
            )

            local_loaders[client_idx]["datasize"] = len(dataidxs)
            local_loaders[client_idx]["train"] = local_train
            # Train set class distribution. Includes class number and # of instances.
            cur_classes = all_targets[dataidxs]
            local_classes = dict(Counter(cur_classes))
            local_loaders[client_idx]["class_distribution"] = local_classes

        else:
            local_loaders[client_idx]["train"], _ = DATA_LOADERS[dataset_name](
                root, mode='tr', batch_size=batch_size, dataidxs=dataidxs, dataset_label=dataset_name
            )
            local_loaders[client_idx]["datasize"] = len(dataidxs)

    if net_dataidx_map_test is not None:
        print(">>> Distributing client test data...")
        if partition.use_val:
            m = 'te'
        else:
            m = 'tr'
        for client_idx, dataidxs in net_dataidx_map_test.items():
            if dataset_name != 'seismic':
                local_testloader = DATA_LOADERS[dataset_name](
                    root, mode=m, batch_size=batch_size, dataidxs=dataidxs, dataset_label=dataset_name
                )

                local_loaders[client_idx]["test"] = local_testloader
                local_loaders[client_idx]["test_size"] = len(dataidxs)
            else:
                local_testloader, gt_labels = DATA_LOADERS[dataset_name](
                    root, mode=m, batch_size=1, dataidxs=dataidxs, dataset_label=dataset_name
                )
                local_loaders[client_idx]["test"] = local_testloader
                local_loaders[client_idx]["test_size"] = gt_labels

    ################################################################################################################
    # Global Dataloader (For testing generalization)
    ###############################################################################################################
    if dataset_name != 'seismic':
        test_global_loader = DATA_LOADERS[dataset_name](root, mode='te', batch_size=batch_size, dataset_label=dataset_name)
        global_loaders = {
            "test": test_global_loader,
            "test_size": int(len(test_global_loader)*batch_size)
        }
    else:
        test_batch_size = 1
        test_global_loader1 = DATA_LOADERS[dataset_name](root, mode='te1', batch_size=test_batch_size, dataset_label=dataset_name)
        test_global_loader2 = DATA_LOADERS[dataset_name](root, mode='te2', batch_size=test_batch_size,
                                                         dataset_label=dataset_name)
        global_loaders = {
            "test1": test_global_loader1,
            "test2": test_global_loader2,
            "test_size1": int(len(test_global_loader1)*test_batch_size),
            "test_size2": int(len(test_global_loader2)*test_batch_size)
        }

    data_distributed = {
        "global": global_loaders,
        "local": local_loaders,
        "num_classes": num_classes,
    }

    return data_distributed


def centralized_partition(all_targets):
    labels = all_targets
    tot_idx = np.arange(len(labels))
    net_dataidx_map = {}

    tot_idx = np.array(tot_idx)
    np.random.shuffle(tot_idx)
    net_dataidx_map[0] = tot_idx

    return net_dataidx_map


def iid_partition(all_targets, n_clients):
    labels = all_targets
    length = int(len(labels) / n_clients)
    tot_idx = np.arange(len(labels))
    net_dataidx_map = {}

    for client_idx in range(n_clients):
        np.random.shuffle(tot_idx)
        data_idxs = tot_idx[:length]
        tot_idx = tot_idx[length:]
        net_dataidx_map[client_idx] = np.array(data_idxs)

    return net_dataidx_map


def create_local(idxs, all_targets, amount=0.20, save_folder='/home/zoe/Dropbox (GhassanGT)/Zoe/InSync/PhDResearch/Code/Results/NeurIPS2024/'):
    # Creating local test clients from partitioned data
    n_clients = len(idxs)
    net_dataidx_test = {i: np.array([], dtype="int64") for i in range(n_clients)}
    for i in range(len(idxs)):
        current_client_idxs = idxs[i]
        local_test_amount = int(len(current_client_idxs)*amount)
        # get unique classes
        classes = all_targets[current_client_idxs]
        unique_classes = np.unique(classes)
        num_classes = len(unique_classes)
        per_class = int(local_test_amount/num_classes)
        # for client's total num of classes, select local test idxs
        test_idxs = np.array([])
        for c in range(len(unique_classes)):
            curr_class = unique_classes[c]
            class_idxs = current_client_idxs[np.where(all_targets[current_client_idxs] == curr_class)]
            try:
                test_idxs_class = np.random.choice(class_idxs, size=per_class, replace=False)
            except ValueError:
                test_idxs_class = np.random.choice(class_idxs, size=int(amount*len(class_idxs)), replace=False)
            test_idxs = np.concatenate((test_idxs, test_idxs_class))
        # if test idxs are still empty, randomly select samples
        if len(test_idxs) == 0:
            test_idxs = np.random.choice(current_client_idxs, size=int(amount*len(current_client_idxs)), replace=False)
        # For each client, get idxs corresponding to local test set
        net_dataidx_test[i] = test_idxs.astype(int)
        new_train = np.where(np.isin(current_client_idxs, test_idxs, invert=True))[0]
        idxs[i] = current_client_idxs[new_train].astype(int)

    np.save(save_folder + 'test_idxs.npy', net_dataidx_test, allow_pickle=True)
    np.save(save_folder + 'train_idxs.npy', idxs, allow_pickle=True)
    return net_dataidx_test, idxs


def sharding_partition(all_targets, n_clients, shard_per_user, rand_set_all=[]):
    # Create empty dictionary, where each key is represented by a client
    net_dataidx_map = {i: np.array([], dtype="int64") for i in range(n_clients)}
    idxs_dict = {}
    num_classes = len(np.unique(all_targets))
    shard_per_class = int(shard_per_user * n_clients / num_classes)
    # for each data sample
    for i in range(len(all_targets)):
        label = torch.tensor(all_targets[i]).item() # get current label
        if label not in idxs_dict.keys(): # if this label hasn't been seen yet
            idxs_dict[label] = [] # init
        idxs_dict[label].append(i) # for each specific class label, record all indexes that correspond to it

    for label in idxs_dict.keys():
        x = idxs_dict[label] # total idxs
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x # edits idxs_dict to list shards per class label

    if len(rand_set_all) == 0:
        # Note: shard_per_user denotes how non-iid setup will be. Ideal iid case if when shard_per_user == number of total classes
        class_arr = np.tile(np.arange(num_classes), shard_per_class)
        random.shuffle(class_arr)
        #rand_set_all = [list(x) for x in np.split(class_arr, n_clients)]
        rand_set_all = [list(x) for x in np.array_split(class_arr, n_clients)]
    print('Rand set all: ', rand_set_all)
    # divide and assign
    for i in range(n_clients):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            # choose one of the shards per class
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            # remove shard from being chosen in future
            rand_set.append(idxs_dict[label].pop(idx))
        net_dataidx_map[i] = np.concatenate(rand_set).astype("int")

    return net_dataidx_map, rand_set_all


def lda_partition(all_targets, n_clients, alpha):
    labels = all_targets
    length = int(len(labels) / n_clients)
    net_dataidx_map = {}

    unique_classes = np.unique(labels)

    tot_idx_by_label = []
    for i in unique_classes:
        idx_by_label = np.where(labels == i)[0]
        tot_idx_by_label.append(idx_by_label)

    min_size = 0

    while min_size < 10:
        idx_batch = [[] for _ in range(n_clients)]
        N, K = len(all_targets), len(np.unique(all_targets))

        for k in range(K):
            # get a list of batch indexes which are belong to label k
            idx_k = np.where(all_targets == k)[0]
            idx_batch, min_size = dirichlet(
                N, alpha, n_clients, idx_batch, idx_k
            )

    for j in range(len(idx_batch)):
        idx_batch[j] = np.array(idx_batch[j]).astype('int')

    for i in range(n_clients):
        np.random.shuffle(idx_batch[i])
        net_dataidx_map[i] = idx_batch[i]

    return net_dataidx_map

def dirichlet(N, alpha, client_num, idx_batch, idx_k):
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array(
        [p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)]
    )
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the batch list for each client
    idx_batch = [
        idx_j + idx.tolist()
        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
    ]
    min_size = min([len(idx_j) for idx_j in idx_batch])

    return idx_batch, min_size

def partition_class_samples_with_dirichlet_distribution(
    alpha, client_num, targets, class_num
):
    net_dataidx_map = {}
    min_size = 0
    min_require_size = 10
    N = len(targets)
    # print(N)
    # print(class_num)
    # print(targets)

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(client_num)]
        for k in np.unique(targets):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, client_num))

            # get the index in idx_k according to the dirichlet distribution
            proportions = np.array(
                [p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            # generate the batch list for each client
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(len(idx_batch)):
        idx_batch[j] = np.array(idx_batch[j]).astype('int')

    for j in range(client_num):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map


def oracle_partition(all_targets, oracle_size=0):
    oracle_idxs = None

    if oracle_size != 0:
        idxs_dict = {}

        for i in range(len(all_targets)):
            label = torch.tensor(all_targets[i]).item()
            if label not in idxs_dict.keys():
                idxs_dict[label] = []
            idxs_dict[label].append(i)

            oracle_idxs = []

        for value in idxs_dict.values():

            oracle_idxs += value[0:oracle_size]

    return oracle_idxs


def get_dist_vec(dataloader, num_classes):
    """Calculate distribution vector for local set"""
    targets = dataloader.dataset.targets
    dist_vec = torch.zeros(num_classes)
    counter = Counter(targets)
    # how frequently the classes appear
    for class_idx, count in counter.items():
        dist_vec[class_idx] = count

    dist_vec /= len(targets)

    return dist_vec


def net_dataidx_map_counter(net_dataidx_map, all_targets):
    data_map = [[] for _ in range(len(net_dataidx_map.keys()))]
    num_classes = len(np.unique(all_targets))

    prev_key = -1
    for key, item in net_dataidx_map.items():
        client_class_count = [0 for _ in range(num_classes)]
        class_elems = all_targets[item]
        for elem in class_elems:
            client_class_count[elem] += 1

        data_map[key] = client_class_count

    return np.array(data_map)
