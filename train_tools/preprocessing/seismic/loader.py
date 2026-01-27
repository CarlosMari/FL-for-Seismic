import numpy as np
import torch.utils.data as data
from torchvision.transforms import transforms

from train_tools.preprocessing.seismic.datasets import InlineLoader


def test_data(root):
    modified_root = root.split('seismic')[0]
    test_seismic2 = np.load(modified_root + 'test_once/test2_seismic.npy')
    test1 = np.load(modified_root + 'test_once/test1_seismic.npy')
    test_seismic2 = (test_seismic2 - test_seismic2.min()) / (test_seismic2.max() - test_seismic2.min())
    test_seismic1 = (test1 - test1.min()) / (test1.max() - test1.min())

    test_labels1 = np.load(modified_root + 'test_once/test1_labels.npy')
    test_labels2 = np.load(modified_root + 'test_once/test2_labels.npy')

    return test_seismic1, test_seismic2, test_labels1, test_labels2

def read_standardize_data(root):
    modified_root = root.split('seismic')[0]
    train_data = np.load(modified_root + 'train/train_seismic.npy')
    train_labels = np.load(modified_root + 'train/train_labels.npy')
    # Normalize seismic
    train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())

    return train_data, train_labels

def create_clients_rand_sections(data, num_clients):
    client_idxs = {}
    choice_tracker = []
    choices=['crossline']

    for c in range(num_clients):
        choice = np.random.choice(choices, size=1)[0]
        if choice == 'inline':
            arr = data.shape[0]
        else:
            arr = data.shape[1]

        amt_missing = 0
        while amt_missing == 0:
            start_pt = np.random.choice(np.arange(arr), size=1, replace=False)[0]
            idx_missing = np.random.choice(np.arange(start_pt, arr), size=1)[0]
            amt_missing = idx_missing - start_pt

        cur_idxs = np.arange(start_pt, idx_missing) # Get idxs of missing (continuous) section
        client_idxs[c] = cur_idxs
        choice_tracker.append(choice)

    return client_idxs

def create_local_test(idxs, amt=0.2):
    net_test = {}
    for c in range(len(idxs)):
        cur_idxs = idxs[c]
        test_length = int(amt * len(cur_idxs))
        start = np.random.choice(cur_idxs, size=1, replace=False)[0]
        start_idx = np.where(cur_idxs == start)[0][0]
        test_idxs = cur_idxs[start_idx:start_idx + test_length]
        new_tr_idxs = cur_idxs[np.where(np.isin(cur_idxs, test_idxs, invert=True))[0]]
        idxs[c] = new_tr_idxs
        net_test[c] = test_idxs
    return idxs, net_test

def get_seismic_idxs(root, num_clients):
    # call func to read and standardize data
    train, train_labels = read_standardize_data(root)
    # Labels
    # Distribute client train and test data
    temp_train = create_clients_rand_sections(train, num_clients)
    train_idxs, test_idxs = create_local_test(temp_train)
    return train_idxs, test_idxs

def get_dataloader_seismic(root, mode='tr', batch_size=4, dataidxs=None, dataset_label=''):

    transform = transforms.Compose([transforms.ToTensor()])

    if mode == 'tr':
        dataset_, labels = read_standardize_data(root)
        gt_labels = labels[:, dataidxs, :]
        dataset = InlineLoader(seismic_cube=dataset_, label_cube=labels, transform=transform,
                               inline_inds=dataidxs)
        # Create overall dataloader
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        return dataloader, gt_labels

    elif mode == 'te1':
        test_seismic1, test_seismic2, test_lab1, test_lab2 = test_data(root)
        dataset = InlineLoader(seismic_cube=test_seismic1, label_cube=test_lab1, transform=transform,
                               train_status=False, inline_inds=list(np.arange(0, test_seismic1.shape[1])))
        # Create overall dataloader
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
    else:
        test_seismic1, test_seismic2, test_lab1, test_lab2 = test_data(root)
        dataset = InlineLoader(seismic_cube=test_seismic2, label_cube=test_lab2, transform=transform,
                               train_status=False, inline_inds=list(np.arange(0, test_seismic2.shape[1])))
        # Create overall dataloader
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

    return dataloader