import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import copy
import time
from timeit import default_timer as timer
from .measures import *
__all__ = ["BaseServer"]

from .measures import model_metrics_glob


class BaseServer:
    def __init__(
        self,
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
        n_rounds=200,
        sample_ratio=0.1,
        local_epochs=5,
        device="cuda:0",
    ):
        """
        Server class controls the overall experiment.
        """
        self.algo_params = algo_params
        self.n_rounds = n_rounds
        self.dynamic_ratio = dynamic_ratio
        self.consistency = consistency
        self.num_classes = data_distributed["num_classes"]

        self.rd = None

        self.model = model
        self.df_glob = pd.DataFrame([])
        self.df_local = pd.DataFrame([])

        self.local_testloaders = data_distributed["local"]
        self.device = device
        self.dataset = dataset

        # Base server sets default test loader as global test set
        if self.dataset != 'seismic':
            # two global loaders for seismic
            self.global_testloader = data_distributed["global"]["test"]

        if self.dataset == 'olives':
            # multilabel classification
            self.criterion = nn.BCEWithLogitsLoss()
            self.multilabel = True
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.multilabel = False

        self.data_distributed = data_distributed
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sample_ratio = sample_ratio
        self.n_clients = len(data_distributed["local"].keys())
        self.local_epochs = local_epochs
        self.save_classwise = {}
        self.save_folder = save_folder

        # Store overall global model performance on local and global test sets (forgetting metrics)
        self.filename = self.save_folder + 'global_stats.npy'
        if stats:
            self.stats = np.load(self.filename, allow_pickle=True).item(0)
            self.sampled_acc = self.stats["sampled_acc"]
            self.global_on_local = self.stats["global_on_local"]
            self.global_client = self.stats["global_client"]
            self.classwise = self.stats["classwise"]

            self.global_on_global = self.stats["global_on_global"]
            self.client_history = self.stats["client_history"]
            self.current_round = self.stats["round"] + 1
            # Spreadsheets
            self.df_glob = pd.read_excel(self.save_folder + 'global_results.xlsx')
            self.df_local = pd.read_csv(self.save_folder + 'global_on_local.csv')
            self.local_clients_info = pd.read_excel(self.save_folder + 'local_results.xlsx')
            self.resume = True
        else:
            self.stats = {}
            self.resume = False
            self.current_round = 0
            self.global_on_global = np.array([])
            self.local_clients_info = pd.DataFrame([])
            self.client_history = {}
            self.classwise = None
            self.sampled_acc = {}
            self.global_on_local = {}
            self.global_client = {}

    def run(self):
        """Run the FL experiment"""
        self._print_start()

        for round_idx in range(self.current_round, self.n_rounds):
            print('Round ' + str(round_idx))
            self.rd = round_idx

            self.stats["round"] = round_idx

            start_time = timer()

            # Make local sets to distributed to clients
            sampled_clients = self._client_sampling(round_idx)
            # modify client history save by associating clients with round sampled
            self.client_history[round_idx] = sampled_clients
            self.stats["client_history"] = self.client_history
            # Client training stage to upload weights & stats
            # For continual learning experiments, 1 round == 1 task out of a sequence
            updated_local_weights, client_sizes, round_results = self._clients_training(
                sampled_clients, round_idx
            )
            print('for all clients, round results: ', round_results)
            # Get aggregated weights & update global
            ag_weights = self._aggregation(updated_local_weights, client_sizes)

            # Update global weights and evaluate statistics
            self.df_glob = self._update_and_evaluate(ag_weights, round_results, round_idx, start_time,
                                                     sampled_clients=sampled_clients, df_global=self.df_glob)
            # End time
            end_time = timer()
            elapsed_time = end_time - start_time
            self.df_glob.at[round_idx, 'time cost'] = elapsed_time
            # Save temporary model
            model_path = os.path.join(self.save_folder, "model.pth")
            torch.save({'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, model_path)

            # save spreadsheet
            self.df_glob.to_excel(self.save_folder + 'global_results.xlsx', index=False)
            self.df_local.to_csv(self.save_folder + 'global_on_local.csv', index=False)
            self.local_clients_info.to_excel(self.save_folder + 'local_results.xlsx', index=False)

    def _clients_training(self, sampled_clients, r_idx):
        """Conduct local training and get trained local models' weights"""

        updated_local_weights, client_sizes = [], []
        round_results = {}

        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        # Client training stage
        for client_idx in sampled_clients:

            # Fetch client datasets
            self._set_client_data(client_idx)
            # Download global
            self.client.download_global(server_weights, server_optimizer)

            # Local training
            local_results, local_size = self.client.train()

            df = pd.DataFrame.from_dict([local_results])
            df['Round'] = r_idx
            self.local_clients_info = pd.concat([self.local_clients_info, df])
            # Upload locals
            updated_local_weights.append(self.client.upload_local())

            # Update results
            round_results = self._results_updater(round_results, local_results)
            client_sizes.append(local_size)

            # Reset local model
            self.client.reset()

        return updated_local_weights, client_sizes, round_results

    def _client_sampling(self, round_idx):
        """Sample clients by given sampling ratio"""

        # make sure for same client sampling for fair comparison
        np.random.seed(round_idx)
        if self.dynamic_ratio:
            if round_idx < int(self.n_rounds / 2):
                ratio = self.sample_ratio[0]
            else:
                ratio = self.sample_ratio[-1]
        else:
            ratio = self.sample_ratio

        clients_per_round = max(int(self.n_clients * ratio), 1)

        if not self.consistency:
            sampled_clients = np.random.choice(
                self.n_clients, clients_per_round, replace=False
            )
        else:
            try:
                if len(self.client_history[round_idx-1]) == clients_per_round:
                    sampled_clients = self.client_history[round_idx-1]
            except KeyError:
                if round_idx == 0:
                    sampled_clients = np.random.choice(
                        self.n_clients, clients_per_round, replace=False
                    )
            else:
                sampled_clients = np.random.choice(
                    self.n_clients, clients_per_round, replace=False
                )

        print('Number of sampled clients: ', len(sampled_clients))
        return sampled_clients

    def _set_client_data(self, client_idx):
        """Assign local client datasets."""
        self.client.datasize = self.data_distributed["local"][client_idx]["datasize"]
        self.client.trainloader = self.data_distributed["local"][client_idx]["train"]
        self.client.local_testloader = self.data_distributed["local"][client_idx]["test"]
        self.client.testsize = self.data_distributed["local"][client_idx]["test_size"]

        self.client.global_testloader = self.data_distributed["global"]["test"]
        self.client.global_test_size = self.data_distributed["global"]["test_size"]
        self.client.round = self.rd
        self.client.current_client = client_idx

    def _aggregation(self, w, ns):
        prop = torch.tensor(ns, dtype=torch.float)
        # make sum equal to 1
        prop /= torch.sum(prop)
        print('------WEIGHTS: ', prop)
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * prop[0]

        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k] * prop[i]

        return copy.deepcopy(w_avg)

    def _results_updater(self, round_results, local_results):
        """Combine local results as clean format"""
        for key, item in local_results.items():
            if key not in round_results.keys():
                round_results[key] = [item]
            else:
                round_results[key].append(item)
        return round_results

    def _print_start(self):
        """Print initial log for experiment"""

        if self.device == "cpu":
            return "cpu"

        if isinstance(self.device, str):
            device_idx = int(self.device[-1])
        elif isinstance(self.device, torch._device):
            device_idx = self.device.index

        device_name = torch.cuda.get_device_name(device_idx)
        print("")
        print("=" * 50)
        print("Train start on device: {}".format(device_name))
        print("=" * 50)

    def _print_stats(self, round_results, test_accs, round_idx, round_elapse):
        print(
            "[Round {}/{}] Elapsed {}s (Current Time: {})".format(
                round_idx + 1,
                self.n_rounds,
                round(round_elapse, 1),
                time.strftime("%H:%M:%S"),
            )
        )

        print("[GLOBAL Stat] Acc on Global Test Set - ", test_accs)

    def _update_and_evaluate(self, ag_weights, round_results, round_idx, start_time, sampled_clients, df_global=pd.DataFrame([])):
        """Evaluate experiment statistics."""

        # Update Global Server Model with Aggregated Model Weights
        self.model.load_state_dict(ag_weights)

        # Measure Accuracy Statistics
        #####################################################################################################
        # GLOBAL
        if len(self.global_on_global) == 0:
            if self.multilabel:
                self.global_on_global = np.zeros(shape=(self.data_distributed["global"]["test_size"], self.num_classes))
            else:
                self.global_on_global = np.zeros(shape=self.data_distributed["global"]["test_size"])

        # get global model performance on global test set
        new_acc, forgets_glob, nfr_glob, glob_test_acc, pos, unchanged, self.nfr_class = model_metrics_glob(self.model,
                                                                                            dataloader=self.global_testloader,
                                                                                            previous_acc=self.global_on_global,
                                                                                            classes=self.num_classes,
                                                                                            dataset=self.dataset)

        print('**')
        print('Classwise NFR: ', self.nfr_class)
        if round_idx > 0:
            # save off info
            np.save(self.save_folder + 'classwise_nfr_' + str(round_idx) + '.npy', self.nfr_class)
        print('**')
        if self.multilabel:
            classwise_glob = np.mean(glob_test_acc['class'])
            glob_test_ = glob_test_acc['macro']
        else:
            self.prev_classwise = self.classwise
            classwise_arr, classwise_glob, self.classwise = evaluate_model_classwise(self.model, dataloader=self.global_testloader,
                                                         num_classes=self.num_classes, device=self.device)
            glob_test_ = glob_test_acc

        self.global_on_global = new_acc # This is an array of 0 and 1's; 1 == sample is correct, 0 == incorrect

        ######################################################################################################

        # # Test Global Model on Each Local Client Test Set
        print('Test Global Model on All Clients')
        backward_transfer = []
        for client in range(self.n_clients):
            current_loader = self.local_testloaders[client]["test"]
            try:
                # prev_loc_acc saves WHICH samples were predicted correctly in the past
                prev_loc_acc = self.global_on_local[client]
            except KeyError:
                if self.multilabel:
                    prev_loc_acc = np.zeros(shape=(self.data_distributed["local"][client]["test_size"], self.num_classes))
                else:
                    prev_loc_acc = np.zeros(shape=self.data_distributed["local"][client]["test_size"])
            # Test global model on each local client test set
            p, cur_forgets, cur_nfr, cur_acc_, cur_pos, cur_un, _ = model_metrics_glob(model=self.model,
                                                                                    dataloader=current_loader,
                                                                                    previous_acc=prev_loc_acc,
                                                                                    classes=self.num_classes,
                                                                                    dataset=self.dataset)
            if self.multilabel:
                cur_acc = cur_acc_['macro']
                classwise_local = np.mean(cur_acc_['class'])
            else:
                _, classwise_local, _ = evaluate_model_classwise(model=self.model, dataloader=current_loader,
                                                              num_classes=self.num_classes, device=self.device)
                cur_acc = cur_acc_
            # Update latest SAVED ACCURACY on client - this is for ALL available clients
            self.global_client[client] = cur_acc
            # pull up past client history
            if round_idx != 0:
                try:
                    client_past_acc = self.sampled_acc[client]
                except KeyError:
                    client_past_acc = -1

                if client_past_acc != -1:
                    backward_transfer.append(cur_acc - client_past_acc)

            # If client has been sampled, specifically record this accuracy (keep this separate from self.global_client, which contains global acc on ALL clients)
            if client in self.client_history[round_idx]:
                self.sampled_acc[client] = cur_acc

            self.global_on_local[client] = p
            local_dict = {"Client": client, 'Global model test acc': cur_acc, 'Global model NFR': cur_nfr,
                          'Forgets': cur_forgets,
                          'Pos flips': cur_pos,
                          'Unchanged': cur_un,
                          'Global model classwise': classwise_local, 'Round': round_idx}
            local_df = pd.DataFrame.from_dict([local_dict])
            self.df_local = pd.concat([self.df_local, local_df], ignore_index=True)

        #  Change learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        #
        round_elapse = time.time() - start_time
        self._print_stats(round_results, glob_test_, round_idx, round_elapse)
        # # save data in excel
        print('--BACKWARD: ', backward_transfer)
        print('Sampled clients: ', self.sampled_acc.keys())
        if len(backward_transfer) != 0:
            backward_transfer = sum(backward_transfer) / len(backward_transfer)
            print('Backward transfer on past sampled clients: ', backward_transfer)
        else:
            backward_transfer = 0
        print("-" * 50)

        # Save all stats
        self.stats["sampled_acc"] = self.sampled_acc
        self.stats["global_on_local"] = self.global_on_local
        self.stats["global_client"] = self.global_client
        self.stats["global_on_global"] = self.global_on_global
        self.stats["classwise"] = self.classwise

        self.save_classwise[round_idx] = self.classwise
        np.save(self.save_folder + 'classwise.npy', self.save_classwise, allow_pickle=True)

        np.save(self.filename, self.stats, allow_pickle=True)

        df_global.at[round_idx, 'Round'] = round_idx
        df_global.at[round_idx, 'Global test acc'] = glob_test_
        df_global.at[round_idx, 'Global classwise accuracy'] = classwise_glob
        df_global.at[round_idx, 'Global nfr'] = nfr_glob
        df_global.at[round_idx, 'Forgets'] = forgets_glob
        df_global.at[round_idx, 'Pos flips'] = pos
        df_global.at[round_idx, 'Unchanged'] = unchanged
        df_global.at[round_idx, 'Backward transfer'] = backward_transfer
        return df_global