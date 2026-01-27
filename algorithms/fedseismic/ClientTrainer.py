import numpy as np
import torch
import os
import sys
import copy

from torch import nn

from algorithms.fedseismic.metrics import eval_model

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, criterion, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """
        self.criterion = criterion
        self.criterion = nn.CrossEntropyLoss()
        self.classifier = None

    def train(self):
        """Local training"""

        # Keep global model's weights
        self._keep_global()

        self.model.train()
        self.model.to(self.device)
        c = nn.MSELoss().to(self.device)

        local_size = 0

        for ep in range(self.local_epochs):
            for i, (img, target) in enumerate(self.trainloader):
                img = img.to(self.device).type(torch.float)
                target = target.to(self.device).type(torch.long)
                output, recon = self.model(img)
                self.optimizer.zero_grad()
                #loss = self.criterion(output, target, dg_logits)
                loss = self.criterion(output, target)
                reconstruction_loss = c(recon, img)
                loss += reconstruction_loss
                loss.backward()
                self.optimizer.step()

                local_size += (i*img.size(0))

        local_results = self._get_local_stats(current_client=self.current_client)

        return local_results, local_size

    def _get_local_stats(self, current_client):
        # eval local models on local test sets
        local_results = {}
        # Set up forgetting counters
        # Local test sets
        try:
            previous_client_acc = self.prev_acc[current_client]
            r = 1
        except KeyError:
            print(self.testsize.shape)
            previous_client_acc = np.zeros(shape=self.testsize.shape)
            r = 0

        # Eval trained model (on the current client) on the local client test set
        new, miou, miou_class, nfr = eval_model(model=copy.deepcopy(self.model), data_loader=self.local_testloader,
                                                previous_acc=previous_client_acc,
                                                round_idx=r, save_file=self.testsize)

        # update saved accuracy for client (LOCAL MODELS --> LOCAL TEST SET)
        self.prev_acc[current_client] = new.astype(int)
        local_results['Sampled Client'] = current_client
        local_results["local on local nfr"] = nfr
        local_results["local on local miou"] = miou
        local_results["local on local classwise"] = miou_class
        local_results["local on local classwise tot"] = np.mean(miou_class)
        return local_results

    def _get_dg_logits(self, data):

        with torch.no_grad():
            dg_logits = self.dg_model(data)

        return dg_logits

    def _prior_local_logits(self, data):

        with torch.no_grad():
            logits = self.prior_local(data)

        return logits

    def upload_local_classifier(self):
        """Uploads local model's classifier"""
        local_classifier = copy.deepcopy(self.model)

        return local_classifier

    def _old_local(self):
        self.prior_local = copy.deepcopy(self.classifier)
        self.prior_local.to(self.device)

        for params in self.prior_local.parameters():
            params.requires_grad = False