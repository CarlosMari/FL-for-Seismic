import copy
import time
import os
import sys
import pandas as pd
import numpy as np
import torch

from algorithms.fedseismic.metrics import eval_model

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseServer import BaseServer
from algorithms.fedseismic.ClientTrainer import ClientTrainer
from algorithms.fedseismic.loss import *

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler, dynamic_ratio, consistency, dataset, save_folder, stats, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, dynamic_ratio, consistency, dataset, save_folder, stats, **kwargs
        )
        local_criterion = self._get_local_criterion(self.algo_params)

        self.client = ClientTrainer(
            local_criterion,
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
            save_folder=self.save_folder,
            dataset=self.dataset,
            local_stats=stats
        )

        self.client_classifiers = {}
        self.prior_history = np.array([])
        self.round = 0
        self.global_testloader1 = data_distributed["global"]["test1"]
        self.global_testloader2 = data_distributed["global"]["test2"]

        self.global_on_global1 = np.array([])
        self.global_on_global2 = np.array([])

        self.test_set1 = np.load('/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/BIGandDATA/Seismic/data/test_once/test1_labels.npy')
        self.test_set2 = np.load('/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/BIGandDATA/Seismic/data/test_once/test2_labels.npy')

        self.df_glob = pd.DataFrame(columns=['Round', 'Global test acc test1', 'Global classwise test1', 'Global test acc test2',
                              'Global classwise test2', 'Global nfr test1', 'Global nfr test2'])
        self.df_glob['Global classwise test1'] = self.df_glob['Global classwise test1'].astype(object)
        self.df_glob['Global classwise test2'] = self.df_glob['Global classwise test2'].astype(object)

        print("\n>>> FedSeismic Server initialized...\n")

    def run(self):
        """Run the FL experiment"""
        self._print_start()

        for round_idx in range(self.n_rounds):
            print('Round ' + str(round_idx))
            self.round = round_idx
            start_time = time.time()

            # Make local sets to distributed to clients
            sampled_clients = self._client_sampling(round_idx)
            # client history
            self.client_history[round_idx] = sampled_clients
            self.stats["client_history"] = self.client_history
            # Double check how many clients are sampled
            print('LENGTH OF SAMPLED CLIENTS: ', len(sampled_clients))
            # Client training stage to upload weights & stats
            updated_local_weights, client_sizes, round_results = self._clients_training(
                sampled_clients, round_idx
            )
            print('for all clients, round results: ', round_results)
            #############################################################
            # Compute average accuracy and NFR
            round_info = self.local_clients_info[self.local_clients_info['Round'] == round_idx]
            accuracies = round_info['local on local classwise tot'].to_numpy()
            custom_weights = []
            # Determine weightage of each client; e.g. by accuracy
            for i in range(len(sampled_clients)):
                wt = accuracies[i]
                custom_weights.append(wt)
            #############################################################

            # Get aggregated weights & update global
            ag_weights = self._aggregation(updated_local_weights, custom_weights)

            # Update global weights and evaluate statistics
            self.df_glob = self._update_and_evaluate(ag_weights, round_results, round_idx, start_time,
                                                     sampled_clients=sampled_clients, df_global=self.df_glob)
            # Save temporary model
            model_path = os.path.join(self.save_folder, "model_" + str(round_idx) + ".pth")
            torch.save(self.model.state_dict(), model_path)
        # save spreadsheet
        self.df_glob.to_excel(self.save_folder + 'global_results.xlsx', index=False)
        #self.df_local.to_excel(self.save_folder + 'global_on_local.xlsx', index=False)
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
            #print('Local results: ', local_results)
            # save local results
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

    def _get_local_criterion(self, algo_params):
        beta = algo_params.beta

        criterion = weighted_loss(beta)

        return criterion

    def _set_client_data(self, client_idx):
        """Assign local client datasets."""
        self.client.datasize = self.data_distributed["local"][client_idx]["datasize"]
        self.client.trainloader = self.data_distributed["local"][client_idx]["train"]
        self.client.global_testloader1 = self.data_distributed["global"]["test1"]
        self.client.global_testloader2 = self.data_distributed["global"]["test2"]
        self.client.local_testloader = self.data_distributed["local"][client_idx]["test"]
        self.client.testsize = self.data_distributed["local"][client_idx]["test_size"]
        print('Current client: ', client_idx)
        self.client.current_client = client_idx

    def _update_and_evaluate(self, ag_weights, round_results, round_idx, start_time, sampled_clients, df_global=pd.DataFrame([])):
        """Evaluate experiment statistics."""

        pred_folder = '/home/zoe/GhassanGT Dropbox/Zoe Fowler/Zoe/InSync/BIGandDATA/Federated_Learning/10-18-24/seismic/10_Clients_fedseismic_0.5/1/preds/'
        # Update Global Server Model with Aggregated Model Weights
        self.model.load_state_dict(ag_weights)

        # Measure Accuracy Statistics
        if len(self.global_on_global1) == 0:
            self.global_on_global1 = np.zeros(shape=self.test_set1.shape)
            self.global_on_global2 = np.zeros(shape=self.test_set2.shape)
        # get global model performance on global test set 1
        preds1, miou1, miou_class1, nfr1 = eval_model(model=copy.deepcopy(self.model), data_loader=self.global_testloader1,
                                                      previous_acc=self.global_on_global1, save_file=self.test_set1,
                                                      round_idx=round_idx)
        np.save(pred_folder + 'testset1_' + str(round_idx) + '_preds.npy', preds1)
        # test set 2
        preds2, miou2, miou_class2, nfr2 = eval_model(model=copy.deepcopy(self.model), data_loader=self.global_testloader2,
                                                      previous_acc=self.global_on_global2, save_file=self.test_set2,
                                                      round_idx=round_idx)
        np.save(pred_folder+'testset2_'+str(round_idx)+'_preds.npy', preds2)

        self.global_on_global1 = preds1
        self.global_on_global2 = preds2

        # # Test Global Model on Each Local Client Test Set
        #print('Test Global Model on All Clients')

        # for client in range(self.n_clients):
        #     current_loader = self.local_testloaders[client]["test"]
        #     try:
        #         # prev_loc_acc saves WHICH samples were predicted correctly in the past
        #         prev_loc_acc = self.global_on_local[client]
        #     except KeyError:
        #         prev_loc_acc = np.zeros(shape=self.data_distributed["local"][client]["test_size"].shape)
        #     # Test global model on each local client test set
        #     cur_preds, cur_miou, cur_miou_class, cur_nfr = eval_model(model=copy.deepcopy(self.model),
        #                                                               data_loader=current_loader,
        #                                                               previous_acc=prev_loc_acc,
        #                                                               save_file=self.data_distributed["local"][client]["test_size"],
        #                                                               round_idx=round_idx)
        #
        #     local_dict = {"Client": client, 'Global model test acc': cur_miou, 'Global model NFR': cur_nfr,
        #                   'Global model classwise': cur_miou_class, 'Round': round_idx}
        #     local_df = pd.DataFrame.from_dict([local_dict])
        #     self.df_local = pd.concat([self.df_local, local_df], ignore_index=True)
        #
        #  Change learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        #
        round_elapse = time.time() - start_time
        #
        self._print_stats(round_results, (miou1+miou2)/2, round_idx, round_elapse)

        print('Sampled clients: ', self.sampled_acc.keys())
        print("-" * 50)

        df_global.at[round_idx, 'Round'] = round_idx
        df_global.at[round_idx, 'Global test acc test1'] = miou1
        #df_global['Global classwise test1'] = df_global['Global classwise test1'].astype(object)
        df_global.at[round_idx, 'Global classwise test1'] = list(miou_class1)
        df_global.at[round_idx, 'Global test acc test2'] = miou2
        #df_global['Global classwise test2'] = df_global['Global classwise test2'].astype(object)
        df_global.at[round_idx, 'Global classwise test2'] = list(miou_class2)
        df_global.at[round_idx, 'Global nfr test1'] = nfr1
        df_global.at[round_idx, 'Global nfr test2'] = nfr2

        return df_global