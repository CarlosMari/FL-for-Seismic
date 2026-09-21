"""Round orchestration for seismic federated learning."""

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from fedseismic.eval.metrics import evaluate, evaluate_loader, score_loader

from .aggregation import aggregate_state_dicts, get_agg_weights, is_bn_key, normalized_entropy
from .client import (
    ClientTrainer,
    FedBNClientTrainer,
    FedProxClientTrainer,
    FedVLSClientTrainer,
    class_frequency,
)
from .fedkper import FedKPerClientTrainer
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
    miou_local_mean: float | None = None
    miou_local_worst: float | None = None
    miou_global_on_local: float | None = None


class Server:
    def __init__(self, model, client_loaders, criterion, config, device=None,
                 client_info=None, model_factory=None, client_trainer_factory=None,
                 test_loaders=None, client_test_loaders=None, rng=None,
                 probe_loader=None, privacy_logger=None):
        self.client_loaders = client_loaders
        self.client_test_loaders = client_test_loaders or []
        self.criterion = criterion
        self.config = config
        self.device = device or config.device
        self.model = model.to(self.device)
        self.client_info = client_info
        self.model_factory = model_factory or (lambda: deepcopy(model))
        self.client_trainer_factory = client_trainer_factory
        self.test_loaders = test_loaders or {}
        self.probe_loader = probe_loader
        self.privacy_logger = privacy_logger
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
                    "fedkper": FedKPerClientTrainer,
                }.get(self.config.algorithm, ClientTrainer)
            kwargs = dict(
                model=self.model_factory(), loader=self.client_loaders[client],
                criterion=self.criterion, device=self.device,
                local_epochs=self.config.local_epochs, lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                optimizer=self.config.optimizer, momentum=self.config.momentum,
            )
            if factory is FedProxClientTrainer:
                kwargs["mu"] = self.config.mu
            if factory is FedVLSClientTrainer:
                frequencies = class_frequency(self.client_loaders[client], self.config.num_classes)
                kwargs["class_freq"] = frequencies
                kwargs["vacant_mask"] = frequencies == 0
                kwargs["lam"] = self.config.lam
            if factory is FedKPerClientTrainer:
                kwargs["lambda_cap"] = self.config.lambda_cap
                kwargs["grad_clip"] = self.config.grad_clip
            self._client_trainers[client] = factory(**kwargs)
        return self._client_trainers[client]

    def _aggregate_strategy(self):
        if self.config.algorithm == "fedkper":
            return "fedkper"
        return self.config.agg_strategy

    def _aggregate(self, states, selected, class_ious=None, models=None,
                   client_train_accs=None, client_histograms=None):
        strategy = self._aggregate_strategy()
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
            client_models=models, test_loader=self._global_loader("test1"),
            test_labels=self._global_labels("test1"), device=self.device,
            client_train_accs=client_train_accs,
            client_histograms=client_histograms,
        )
        return aggregate_state_dicts(states, weights)

    def _probe_loader(self):
        if self.probe_loader is not None:
            return self.probe_loader
        return self._global_loader("test1")

    def _fedkper_histogram(self, client, model):
        mode = self.config.fedkper_diversity
        if mode == "oracle":
            return np.asarray(self.client_info[client]["class_fracs"], dtype=np.float64)
        if mode == "upload":
            return np.asarray(
                class_frequency(self.client_loaders[client], self.config.num_classes),
                dtype=np.float64,
            )
        if mode == "infer":
            from fedseismic.privacy.estimators import softmax_prior

            probe = self._probe_loader()
            if probe is None:
                raise ValueError("fedkper_diversity='infer' requires a public probe loader")
            return softmax_prior(
                model, probe, self.config.num_classes, self.device,
                self.config.privacy_probe_batches,
            )
        raise ValueError(f"unknown fedkper_diversity {mode!r}")

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
        train_accs = []
        histograms = []
        local_mious = []
        last_round = round_index + 1 >= self.config.num_rounds
        for client in selected:
            trainer = self._trainer(client)
            trainer.download(global_state)
            trainer.train(global_state=global_state)
            uploaded = trainer.upload()
            states.append(uploaded)
            local_models.append(trainer.model)
            train_acc = None
            protocol_hist = None
            if self._aggregate_strategy() == "fedkper" or self.privacy_logger is not None:
                train_acc = evaluate_loader(
                    trainer.model, self.client_loaders[client],
                    self.config.num_classes, self.device,
                )[2]
            if self._aggregate_strategy() == "fedkper":
                train_accs.append(train_acc)
                protocol_hist = self._fedkper_histogram(client, trainer.model)
                histograms.append(protocol_hist)
            if self.config.agg_strategy in {"rare_miou", "invfreq_miou", "invfreq_invmiou"}:
                class_ious.append(self._local_class_iou(trainer.model,
                                                        self.client_loaders[client]))
            if client < len(self.client_test_loaders) and len(self.client_test_loaders[client].dataset):
                local_mious.append(score_loader(
                    trainer.model, self.client_test_loaders[client],
                    self.config.num_classes, self.device, self.config.task,
                )[0])
            if self.privacy_logger is not None:
                scalars = {
                    "algorithm": self.config.algorithm,
                    "train_acc": None if train_acc is None else float(train_acc),
                    "fedkper_diversity": self.config.fedkper_diversity,
                }
                if protocol_hist is not None:
                    scalars["protocol_histogram"] = np.asarray(protocol_hist, dtype=np.float64).tolist()
                    scalars["protocol_entropy"] = normalized_entropy(protocol_hist)
                self.privacy_logger.log_upload(
                    round_index + 1, client, uploaded, global_state,
                    scalars=scalars, is_last_round=last_round,
                )
            trainer.reset()
        self.model.load_state_dict(self._aggregate(
            states, selected, class_ious=class_ious or None, models=local_models,
            client_train_accs=train_accs or None,
            client_histograms=histograms or None,
        ))
        record = RoundRecord(round=round_index + 1, selected_clients=selected)
        if local_mious:
            record.miou_local_mean = float(np.mean(local_mious))
            record.miou_local_worst = float(np.min(local_mious))
        if "test1" in self.test_loaders or "test2" in self.test_loaders:
            per_class_values = []
            for name in ("test1", "test2"):
                if name not in self.test_loaders:
                    continue
                score, per_class = self._eval_global_entry(self.test_loaders[name])
                if name == "test1":
                    record.miou_test1 = score
                else:
                    record.miou_test2 = score
                per_class_values.append(per_class)
            if per_class_values:
                record.per_class_iou = np.mean(per_class_values, axis=0).tolist()
            values = [value for value in (record.miou_test1, record.miou_test2)
                      if value is not None]
            record.miou_final = float(np.mean(values)) if values else None
            if "validation" in self.test_loaders:
                record.miou_validation = self._eval_global_entry(
                    self.test_loaders["validation"],
                )[0]
        self.history.append(record)
        return record

    def evaluate_personalized_local(self):
        """Score every client that has trained, using its last local weights."""
        scores = []
        for client, trainer in self._client_trainers.items():
            if client >= len(self.client_test_loaders):
                continue
            loader = self.client_test_loaders[client]
            if loader is None or not len(loader.dataset):
                continue
            scores.append(score_loader(
                trainer.model, loader, self.config.num_classes, self.device, self.config.task,
            )[0])
        if not scores:
            return None, None
        return float(np.mean(scores)), float(np.min(scores))

    def evaluate_global_on_local(self):
        scores = []
        for loader in self.client_test_loaders:
            if loader is None or not len(loader.dataset):
                continue
            scores.append(score_loader(
                self.model, loader, self.config.num_classes, self.device, self.config.task,
            )[0])
        if not scores:
            return None
        return float(np.mean(scores))

    def _eval_global_entry(self, entry):
        if isinstance(entry, tuple) and entry[1] is not None:
            loader, labels = entry
            return evaluate(self.model, loader, labels, self.device)
        loader = entry[0] if isinstance(entry, tuple) else entry
        score, per_class, _ = score_loader(
            self.model, loader, self.config.num_classes, self.device, self.config.task,
        )
        return score, per_class

    def _local_class_iou(self, model, loader):
        _, per_class, _ = evaluate_loader(
            model, loader, self.config.num_classes, self.device,
        )
        return np.asarray(per_class, dtype=np.float64)

    def _global_loader(self, name):
        entry = self.test_loaders.get(name)
        if entry is None:
            return None
        return entry[0] if isinstance(entry, tuple) else entry

    def _global_labels(self, name):
        entry = self.test_loaders.get(name)
        if isinstance(entry, tuple):
            return entry[1]
        return None

    def run(self):
        for round_index in range(self.config.num_rounds):
            self.run_round(round_index)
        if self.history:
            self.history[-1].miou_global_on_local = self.evaluate_global_on_local()
            local_mean, local_worst = self.evaluate_personalized_local()
            if local_mean is not None:
                self.history[-1].miou_local_mean = local_mean
                self.history[-1].miou_local_worst = local_worst
        if self.privacy_logger is not None:
            self.privacy_logger.write_global(self.model.state_dict())
        return self.history
