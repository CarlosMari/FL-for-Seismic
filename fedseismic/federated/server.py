"""Round orchestration for seismic federated learning."""

from copy import deepcopy
from dataclasses import dataclass, field
import warnings

import numpy as np

from .aggregation import AGGREGATORS, aggregate_state_dicts, get_agg_weights, is_bn_key
from .client import (
    ClientTrainer,
    FedBNClientTrainer,
    FedProxClientTrainer,
    FedVLSClientTrainer,
    class_frequency,
)
from .sampling import sample_clients


@dataclass
class RoundRecord:
    round: int
    selected_clients: list[int]
    miou_test1: float | None = None
    miou_test2: float | None = None
    miou_validation: float | None = None
    miou_final: float | None = None
    per_class_iou: list[float] = field(default_factory=list)


class Server:
    def __init__(self, model, client_loaders, criterion, config, device=None,
                 client_info=None, model_factory=None, client_trainer_factory=None,
                 test_loaders=None, rng=None):
        self.model = model
        self.client_loaders = client_loaders
        self.criterion = criterion
        self.config = config
        self.device = device or config.device
        self.client_info = client_info
        self.model_factory = model_factory or (lambda: deepcopy(model))
        self.client_trainer_factory = client_trainer_factory
        self.test_loaders = test_loaders or {}
        self.rng = rng or np.random.RandomState(config.seed)
        self.history = []
        self._client_trainers = {}

    def _trainer(self, client):
        if client not in self._client_trainers:
            factory = self.client_trainer_factory
            if factory is None:
                factory = {
                    "fedprox": FedProxClientTrainer,
                    "fedbn": FedBNClientTrainer,
                    "fedvls": FedVLSClientTrainer,
                }.get(self.config.algorithm, ClientTrainer)
            kwargs = dict(
                model=self.model_factory(), loader=self.client_loaders[client],
                criterion=self.criterion, device=self.device,
                local_epochs=self.config.local_epochs, lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
            if factory is FedProxClientTrainer:
                kwargs["mu"] = self.config.mu
            if factory is FedVLSClientTrainer:
                frequencies = class_frequency(self.client_loaders[client], self.config.num_classes)
                kwargs["class_freq"] = frequencies
                kwargs["vacant_mask"] = frequencies == 0
                kwargs["lam"] = self.config.lam
            self._client_trainers[client] = factory(**kwargs)
        return self._client_trainers[client]

    def _aggregate(self, states, selected, class_ious=None, models=None):
        strategy = self.config.agg_strategy
        if strategy == "equal":
            if self.config.algorithm == "fedbn":
                averaged = deepcopy(states[0])
                for key in averaged:
                    if is_bn_key(key):
                        continue
                    averaged[key] = sum(state[key] for state in states) / len(states)
                return averaged
            return aggregate_state_dicts(states)
        weights = get_agg_weights(
            strategy, selected, self.client_info, client_class_ious=class_ious,
            client_models=models, test_loader=self.test_loaders.get("test1", (None, None))[0],
            test_labels=self.test_loaders.get("test1", (None, None))[1], device=self.device,
        )
        return aggregate_state_dicts(states, weights)

    def run_round(self, round_index):
        global_state = deepcopy(self.model.state_dict())
        selected = sample_clients(
            self.config.num_clients, self.config.sample_ratio, self.rng,
            self.client_info, self.config.force_rare_client,
        )
        # Legacy FedVLS constructs a frozen model before constructing clients;
        # preserve that RNG consumption even though trainers use a deep copy.
        if self.config.algorithm == "fedvls":
            self.model_factory()
        states = []
        local_models = []
        class_ious = []
        for client in selected:
            trainer = self._trainer(client)
            trainer.download(global_state)
            trainer.train(global_state=global_state)
            states.append(trainer.upload())
            local_models.append(trainer.model)
            if self.config.agg_strategy in {"rare_miou", "invfreq_miou", "invfreq_invmiou"}:
                class_ious.append(self._local_class_iou(trainer.model,
                                                        self.client_loaders[client]))
            trainer.reset()
        self.model.load_state_dict(self._aggregate(
            states, selected, class_ious=class_ious or None, models=local_models,
        ))
        record = RoundRecord(round=round_index + 1, selected_clients=selected)
        if "test1" in self.test_loaders or "test2" in self.test_loaders:
            from fedseismic.eval.metrics import evaluate
            per_class_values = []
            for name in ("test1", "test2"):
                if name in self.test_loaders:
                    loader, labels = self.test_loaders[name]
                    miou, per_class = evaluate(self.model, loader, labels, self.device)
                    if name == "test1":
                        record.miou_test1 = miou
                    else:
                        record.miou_test2 = miou
                    per_class_values.append(per_class)
            if per_class_values:
                record.per_class_iou = np.mean(per_class_values, axis=0).tolist()
            values = [value for value in (record.miou_test1, record.miou_test2)
                      if value is not None]
            record.miou_final = float(np.mean(values)) if values else None
            if "validation" in self.test_loaders:
                loader, labels = self.test_loaders["validation"]
                record.miou_validation = evaluate(
                    self.model, loader, labels, self.device,
                )[0]
        self.history.append(record)
        return record

    def _local_class_iou(self, model, loader):
        from sklearn.metrics import jaccard_score
        import torch

        predictions = []
        targets = []
        model.eval()
        with torch.no_grad():
            for images, labels, _ in loader:
                logits = model(images.to(self.device, dtype=torch.float))
                logits = logits[0] if isinstance(logits, (tuple, list)) else logits
                predictions.append(logits.argmax(dim=1).cpu().numpy().ravel())
                targets.append(labels.numpy().ravel())
        if not predictions:
            return np.zeros(self.config.num_classes, dtype=np.float64)
        return np.asarray(jaccard_score(
            np.concatenate(targets), np.concatenate(predictions),
            labels=list(range(self.config.num_classes)), average=None, zero_division=0,
        ), dtype=np.float64)

    def run(self):
        for round_index in range(self.config.num_rounds):
            self.run_round(round_index)
        return self.history
