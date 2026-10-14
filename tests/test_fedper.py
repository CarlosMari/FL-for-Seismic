"""FedPer keeps the classifier on the client and averages only the base."""

import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fedseismic.config import RunConfig
from fedseismic.federated.fedper import FedPerClientTrainer, personalization_keys
from fedseismic.federated.server import Server


class _Rows(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], self.targets[index], index


class _SplitNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, num_classes)

    def forward(self, x):
        return self.classifier(torch.relu(self.encoder(x.view(x.size(0), -1))))


class _OutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(1, 2, kernel_size=1)
        self.outc = nn.Conv2d(2, 3, kernel_size=1)
        self.reconstruct = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        return self.outc(features), self.reconstruct(features)


def _loader(targets):
    images = torch.randn(len(targets), 4)
    return DataLoader(_Rows(images, torch.tensor(targets)), batch_size=4, shuffle=False)


class PersonalizationKeyTests(unittest.TestCase):
    def test_classifier_module_is_personal(self):
        keys = personalization_keys(_SplitNet().state_dict())
        self.assertEqual(set(keys), {"classifier.weight", "classifier.bias"})

    def test_unet_class_head_is_personal_and_reconstruct_is_not(self):
        keys = personalization_keys(_OutNet().state_dict())
        self.assertEqual(set(keys), {"outc.weight", "outc.bias"})


class FedPerTrainerTests(unittest.TestCase):
    def test_upload_keeps_global_head_and_personal_state_keeps_local_head(self):
        torch.manual_seed(0)
        model = _SplitNet()
        global_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        trainer = FedPerClientTrainer(
            model, _loader([0, 0, 0, 0, 1, 1, 1, 1]), nn.CrossEntropyLoss(),
            device="cpu", local_epochs=3, lr=0.5, optimizer="sgd", weight_decay=0.0,
        )
        trainer.download(global_state)
        trainer.train(global_state=global_state)
        uploaded = trainer.upload()

        for key in ("classifier.weight", "classifier.bias"):
            self.assertTrue(torch.equal(uploaded[key], global_state[key]))
            self.assertFalse(torch.equal(trainer.personal_state[key], global_state[key]))
        self.assertFalse(torch.equal(uploaded["encoder.weight"], global_state["encoder.weight"]))
        self.assertTrue(torch.equal(trainer.model.state_dict()["classifier.weight"], global_state["classifier.weight"]))

    def test_next_round_restores_personal_head_onto_new_base(self):
        torch.manual_seed(0)
        model = _SplitNet()
        global_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        trainer = FedPerClientTrainer(
            model, _loader([0, 0, 0, 0]), nn.CrossEntropyLoss(),
            device="cpu", local_epochs=2, lr=0.5, optimizer="sgd", weight_decay=0.0,
        )
        trainer.download(global_state)
        trainer.train(global_state=global_state)
        head = {key: trainer.personal_state[key].clone() for key in ("classifier.weight", "classifier.bias")}

        next_global = {key: value.detach().clone() for key, value in global_state.items()}
        next_global["encoder.weight"] = next_global["encoder.weight"] + 1
        trainer.download(next_global)
        current = trainer.model.state_dict()
        self.assertTrue(torch.equal(current["encoder.weight"], next_global["encoder.weight"]))
        self.assertTrue(torch.equal(current["classifier.weight"], head["classifier.weight"]))
        self.assertTrue(torch.equal(current["classifier.bias"], head["classifier.bias"]))


class FedPerServerTests(unittest.TestCase):
    def test_round_averages_base_and_leaves_global_classifier_unchanged(self):
        torch.manual_seed(0)
        loaders = [_loader([0, 0, 0, 0, 0, 0, 0, 0]), _loader([2, 2, 2, 2, 2, 2, 2, 2])]
        tests = loaders
        factory = _SplitNet
        initial = factory()
        initial_head = initial.classifier.weight.detach().clone()
        initial_base = initial.encoder.weight.detach().clone()
        cfg = RunConfig(
            dataset="bloodmnist", num_clients=2, num_rounds=1, local_epochs=3,
            batch_size=4, num_classes=3, sample_ratio=1.0, split="iid",
            algorithm="fedper", agg_strategy="equal", device="cpu",
            lr=0.5, optimizer="sgd", weight_decay=0.0,
        )
        server = Server(
            model=initial, client_loaders=loaders, criterion=nn.CrossEntropyLoss(),
            config=cfg, model_factory=factory, client_test_loaders=tests,
        )
        server.run_round(0)
        self.assertTrue(torch.equal(server.model.classifier.weight, initial_head))
        self.assertFalse(torch.equal(server.model.encoder.weight, initial_base))
        heads = [
            trainer.personal_state["classifier.weight"]
            for trainer in server._client_trainers.values()
        ]
        self.assertFalse(torch.equal(heads[0], heads[1]))
        self.assertFalse(torch.equal(heads[0], initial_head))


if __name__ == "__main__":
    unittest.main()
