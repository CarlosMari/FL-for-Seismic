import numpy as np
import torch
import torch.nn as nn
import copy
from .measures import model_metrics
from .measures import *

__all__ = ["BaseClientTrainer"]


class BaseClientTrainer:
    def __init__(self, algo_params, model, local_epochs, device, num_classes, save_folder,
                 dataset='olives', temp_trainloader=None, local_stats=None):
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """
        # Params
        self.local_epochs = local_epochs
        self.device = device
        self.datasize = None
        self.num_classes = num_classes
        self.class_distribution = {}

        # algorithm-specific parameters
        self.algo_params = algo_params
        self.round = None

        # model & optimizer
        self.model = model
        self.lr = 0
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr)

        self.dataset = dataset
        if self.dataset == 'olives':
            self.criterion = nn.BCEWithLogitsLoss()
            self.multilabel = True
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.multilabel = False

        self.trainloader = temp_trainloader
        self.global_testloader = None
        self.local_testloader = None
        self.current_client = None
        self.testsize = None
        self.global_test_size = 0
        # Additional params
        self.named_params = None
        self.global_other_params = None
        self.params_type = None
        self.layer_generator_dict = {}

        # Saved stats
        self.filename = save_folder + '/local_stats.npy'
        if local_stats:
            self.stats = np.load(self.filename, allow_pickle=True).item(0)
            self.prev_acc = self.stats["local_acc"]
            self.glob_prev_acc = self.stats["global_prev_acc"]
        else:
            self.prev_acc = {}
            self.glob_prev_acc = {}
            self.stats = {}

    def train(self):
        """Local training"""
        # Before training, store what global model is and get current client class distribution
        self.seen = list(self.class_distribution.keys())
        self._keep_global()
        local_size = self.datasize
        # Prototype storing
        self.agg_protos_label = {}
        # Train local model
        self.model.train()
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        for ep in range(self.local_epochs):
            local_probs = np.empty((0, self.num_classes), dtype=np.float32)
            global_probs = np.empty((0, self.num_classes), dtype=np.float32)
            for data, targets, _ in self.trainloader:
                self.optimizer.zero_grad()
                # forward pass
                if self.dataset == 'olives':
                    targets = targets.float()

                data, targets = data.to(self.device), targets.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, targets)
                # backward pass
                loss.backward()
                self.optimizer.step()

        # used trained model to get local TRAIN accuracy and TEST accuracy
        local_results = self._get_local_stats(current_client=self.current_client)
        return local_results, local_size


    def _get_local_stats(self, current_client):
        # This typically evaluates local models on global test sets
        local_results = {}
        # local results train accuracy is training accuracy / f1 score for each individual client
        if self.multilabel:
            l = evaluate_model(
                self.model, self.trainloader, self.dataset, self.device
            )
            local_results["train_acc"] = l['macro']
        else:
            local_results["train_acc"] = evaluate_model(
                self.model, self.trainloader, self.dataset, self.device
            )
        # Set up forgetting counters
        # Local test sets
        try:
            previous_client_acc = self.prev_acc[current_client]
        except KeyError:
            if self.multilabel:
                previous_client_acc = np.zeros(shape=(self.testsize, self.num_classes))
            else:
                previous_client_acc = np.zeros(shape=self.testsize)
        # For local model --> tested on global test set
        try:
            previous_global_acc = self.glob_prev_acc[current_client]
        except KeyError:
            if self.multilabel:
                # Shape should be different for multilabel
                previous_global_acc = np.zeros(shape=(self.global_test_size, self.num_classes))
            else:
                previous_global_acc = np.zeros(shape=self.global_test_size)

        # Eval trained model (on the current client) on the global test set
        new_glob, forgets_glob, nfr_glob, global_acc = model_metrics(self.model, self.global_testloader,
                                                     previous_acc=previous_global_acc, dataset=self.dataset)
        if self.multilabel:
            local_on_glob_classwise = np.mean(global_acc['class'])
            glob_acc = global_acc['macro']
        else:
            _, local_on_glob_classwise, _ = evaluate_model_classwise(model=self.model, dataloader=self.global_testloader,
                                                                  device=self.device, num_classes=self.num_classes)
            glob_acc = global_acc

        ######################################################################################################
        new, forgets_local, nfr_local, local_acc = model_metrics(self.model, self.local_testloader,
                                                                 previous_acc=previous_client_acc,
                                                                 dataset=self.dataset)
        if self.multilabel:
            loc_on_loc_classwise = np.mean(local_acc['class'])
            ll = loc_on_loc_classwise
            loc_acc = local_acc['macro']
        else:
            _, loc_on_loc_classwise, ll = evaluate_model_classwise(model=self.model, dataloader=self.local_testloader,
                                                               num_classes=self.num_classes, device=self.device)
            full_classwise = np.copy(ll)
            ll = np.mean(ll)
            loc_acc = local_acc

        # update saved accuracy for client (LOCAL MODELS --> LOCAL TEST AND GLOBAL TEST SET)
        self.prev_acc[current_client] = new.astype(int)
        self.glob_prev_acc[current_client] = new_glob.astype(int)
        # Save info
        self.stats["local_acc"] = self.prev_acc
        self.stats["global_prev_acc"] = self.glob_prev_acc
        np.save(self.filename, self.stats, allow_pickle=True)
        # Convert results to DF
        local_results['Sampled Client'] = current_client
        local_results["local on local nfr"] = nfr_local
        local_results["local on global nfr"] = nfr_glob
        local_results["local on global test acc"] = glob_acc
        local_results["local on local test acc"] = loc_acc
        local_results["local on global classwise"] = local_on_glob_classwise
        local_results["local on local classwise"] = loc_on_loc_classwise
        local_results["local on local classwise tot"] = ll

        local_results['Class Distribution'] = self.seen
        local_results['Class Distribution Len'] = len(self.seen)
        amts = np.zeros(shape=self.num_classes)
        for c in self.class_distribution:
            amts[c] = self.class_distribution[c]
        local_results['Class Amount'] = list(amts)
        local_results['Full Classwise'] = list(full_classwise)

        return local_results

    def download_global(self, server_weights, server_optimizer):
        """Load model & Optimizer"""
        self.model.load_state_dict(server_weights)
        self.optimizer.load_state_dict(server_optimizer)

    def upload_local(self):
        """Uploads local model's parameters"""
        local_weights = copy.deepcopy(self.model.state_dict())

        return local_weights

    def reset(self):
        """Clean existing setups"""
        self.datasize = None
        self.trainloader = None
        self.global_testloader = None
        self.local_testloader = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        self.current_client = None
        self.testsize = None
        self.global_test_size = 0

    def _keep_global(self):
        """Keep distributed global model's weight"""
        self.dg_model = copy.deepcopy(self.model)
        self.dg_model.to(self.device)

        for params in self.dg_model.parameters():
            params.requires_grad = False
