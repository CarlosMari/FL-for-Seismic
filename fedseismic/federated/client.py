"""Client-side training for segmentation FL algorithms."""

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F


def _logits(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _with_log_prior(logits, log_prior):
    if log_prior is None:
        return logits
    if logits.ndim == 4:
        return logits + log_prior.view(1, -1, 1, 1)
    return logits + log_prior.view(1, -1)


class ClientTrainer:
    """Download global weights, train locally, upload weights, then reset."""

    def __init__(self, model, loader, criterion, device="cpu", local_epochs=1,
                 lr=1e-3, weight_decay=1e-4, optimizer="adamw", momentum=0.9,
                 num_classes=None, logit_adjust=False, prior_smooth=0.0,
                 class_reweight=False, public_loader=None):
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = str(optimizer).strip().lower()
        self.momentum = momentum
        self.num_classes = num_classes
        self.logit_adjust = bool(logit_adjust)
        self.prior_smooth = float(prior_smooth)
        self.class_reweight = bool(class_reweight)
        self.public_loader = public_loader
        self.optimizer = None

    def _make_optimizer(self):
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
            )
        raise ValueError(f"unknown optimizer {self.optimizer_name!r}")

    def download(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        return self

    def _log_prior(self):
        if not self.logit_adjust or not self.num_classes:
            return None
        pi = class_frequency(self.loader, self.num_classes)
        smooth = self.prior_smooth
        if smooth:
            pi = (1.0 - smooth) * pi + smooth / float(self.num_classes)
        return torch.as_tensor(
            np.log(np.clip(pi, 1e-8, None)), device=self.device, dtype=torch.float,
        )

    def _class_counts(self, loader):
        if loader is None or not self.num_classes:
            return np.zeros(self.num_classes or 0, dtype=np.float64)
        labels = _dataset_labels(getattr(loader, "dataset", None))
        if labels is None:
            return class_frequency(loader, self.num_classes) * max(len(loader.dataset), 1)
        return np.bincount(np.asarray(labels).astype(int), minlength=self.num_classes).astype(np.float64)

    def _loss_weight(self):
        """Inverse-frequency weights over the examples this client actually trains on."""
        if not self.class_reweight or not self.num_classes:
            return None
        counts = self._class_counts(self.loader)
        if self.public_loader is not None:
            counts = counts + self._class_counts(self.public_loader)
        present = counts > 0
        weights = np.zeros(self.num_classes, dtype=np.float64)
        if present.any():
            weights[present] = counts[present].sum() / (float(present.sum()) * counts[present])
        return torch.as_tensor(weights, device=self.device, dtype=torch.float)

    def _train_loaders(self):
        yield self.loader
        if self.public_loader is not None:
            yield self.public_loader

    def train(self, global_state=None):
        self.model.train()
        self.model.to(self.device)
        self.optimizer = self._make_optimizer()
        log_prior = self._log_prior()
        weight = self._loss_weight()
        for _ in range(self.local_epochs):
            for loader in self._train_loaders():
                for images, targets, _ in loader:
                    images = images.to(self.device, dtype=torch.float)
                    targets = targets.to(self.device, dtype=torch.long)
                    logits = _with_log_prior(_logits(self.model(images)), log_prior)
                    if weight is None:
                        loss = self.criterion(logits, targets)
                    else:
                        loss = F.cross_entropy(logits, targets, weight=weight)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        return self

    def upload(self):
        return deepcopy(self.model.state_dict())

    def reset(self):
        self.optimizer = None
        self.model.zero_grad(set_to_none=True)
        return self


class StepClientTrainer(ClientTrainer):
    """A fixed number of full-dataset steps, not epochs of minibatches.

    ``example`` is the usual mean over examples, so a class contributes in
    proportion to its count. ``class`` is the mean of the per-class mean
    gradients, over the classes this client holds. The class prior is not
    added in the loss. The server adds it at local evaluation when asked.
    """

    def __init__(self, *args, local_steps=1, gradient_balance="example", **kwargs):
        super().__init__(*args, **kwargs)
        if int(local_steps) < 1:
            raise ValueError("local_steps must be positive")
        if gradient_balance not in {"example", "class"}:
            raise ValueError("gradient_balance must be 'example' or 'class'")
        self.local_steps = int(local_steps)
        self.gradient_balance = gradient_balance

    def train(self, global_state=None):
        self.model.train()
        self.model.to(self.device)
        self.optimizer = self._make_optimizer()
        for _ in range(self.local_steps):
            if self.gradient_balance == "class":
                self._class_step()
            else:
                self._example_step()
        return self

    def _example_step(self):
        seen = self._mean_backward(None)
        if seen:
            self.optimizer.step()

    def _class_step(self):
        present = np.flatnonzero(self._class_counts(self.loader) > 0)
        if len(present) == 0:
            return
        accumulated = None
        for class_index in present:
            seen = self._mean_backward(int(class_index))
            if not seen:
                continue
            grads = [parameter.grad.detach().clone() for parameter in self.model.parameters()]
            if accumulated is None:
                accumulated = grads
            else:
                accumulated = [left + right for left, right in zip(accumulated, grads)]
        if accumulated is None:
            return
        scale = 1.0 / float(len(present))
        for parameter, grad in zip(self.model.parameters(), accumulated):
            parameter.grad = grad * scale
        self.optimizer.step()

    def _mean_backward(self, class_index):
        """Mean gradient of one class, or of every example when ``class_index`` is None."""
        self.optimizer.zero_grad(set_to_none=True)
        seen = 0
        for images, targets, _ in self.loader:
            if class_index is not None:
                mask = targets == class_index
                if not bool(mask.any()):
                    continue
                images = images[mask]
                targets = targets[mask]
            images = images.to(self.device, dtype=torch.float)
            targets = targets.to(self.device, dtype=torch.long)
            loss = F.cross_entropy(_logits(self.model(images)), targets, reduction="sum")
            loss.backward()
            seen += int(targets.shape[0])
        if seen:
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(seen)
        return seen


class FedProxClientTrainer(ClientTrainer):
    def __init__(self, *args, mu=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu

    def train(self, global_state=None):
        if global_state is None:
            global_state = deepcopy(self.model.state_dict())
        global_params = [
            global_state[name].detach().to(self.device)
            for name, _ in self.model.named_parameters()
        ]
        self.model.train()
        self.model.to(self.device)
        self.optimizer = self._make_optimizer()
        for _ in range(self.local_epochs):
            for images, targets, _ in self.loader:
                images = images.to(self.device, dtype=torch.float)
                targets = targets.to(self.device, dtype=torch.long)
                loss = self.criterion(_logits(self.model(images)), targets)
                proximal = torch.zeros((), device=self.device)
                for parameter, global_parameter in zip(self.model.parameters(), global_params):
                    proximal = proximal + (parameter - global_parameter).norm(2) ** 2
                loss = loss + (self.mu / 2.0) * proximal
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self


class DittoClientTrainer(ClientTrainer):
    """Personalized model stays on the client. The upload is plain local SGD.

    Li et al., ICLR 2021. The personal weights minimize cross-entropy plus a
    penalty toward the current global model, and they are never uploaded.
    """

    def __init__(self, *args, ditto_lambda=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ditto_lambda = ditto_lambda
        self.personal_state = None
        self._upload_state = None

    def train(self, global_state=None):
        if global_state is None:
            global_state = deepcopy(self.model.state_dict())
        self.download(global_state)
        super().train(global_state=global_state)
        self._upload_state = deepcopy(self.model.state_dict())

        if self.personal_state is None:
            self.download(global_state)
        else:
            self.download(self.personal_state)
        global_params = [
            global_state[name].detach().to(self.device)
            for name, _ in self.model.named_parameters()
        ]
        self.model.train()
        self.optimizer = self._make_optimizer()
        for _ in range(self.local_epochs):
            for images, targets, _ in self.loader:
                images = images.to(self.device, dtype=torch.float)
                targets = targets.to(self.device, dtype=torch.long)
                loss = self.criterion(_logits(self.model(images)), targets)
                proximal = torch.zeros((), device=self.device)
                for parameter, global_parameter in zip(self.model.parameters(), global_params):
                    proximal = proximal + (parameter - global_parameter).norm(2) ** 2
                loss = loss + (self.ditto_lambda / 2.0) * proximal
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.personal_state = deepcopy(self.model.state_dict())
        self.download(self._upload_state)
        return self

    def upload(self):
        if self._upload_state is None:
            return super().upload()
        return deepcopy(self._upload_state)


class FedBNClientTrainer(ClientTrainer):
    """Keep each client's BatchNorm parameters and running statistics local."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bn_state = {
            key: value.detach().clone() for key, value in self.model.state_dict().items()
            if _is_bn_key(key)
        }

    def download(self, state_dict):
        super().download(state_dict)
        self.model.load_state_dict(self.bn_state, strict=False)
        return self

    def upload(self):
        state = super().upload()
        self.bn_state = {
            key: value.detach().clone() for key, value in state.items() if _is_bn_key(key)
        }
        return state


def _is_bn_key(key):
    parts = key.split(".")
    return any(
        part == "double_conv" and index + 1 < len(parts) and parts[index + 1] in ("1", "4")
        for index, part in enumerate(parts)
    )


def calibrated_cross_entropy(logits, targets, class_freq):
    log_freq = torch.log(class_freq + 1e-10)
    return F.cross_entropy(logits + log_freq[None, :, None, None], targets)


def vacant_class_distillation(logits_local, logits_global, vacant_mask):
    if not vacant_mask.any():
        return torch.tensor(0.0, device=logits_local.device)
    vc_idx = vacant_mask.nonzero(as_tuple=True)[0]
    local_log_probs = F.log_softmax(logits_local, dim=1)
    global_probs = F.softmax(logits_global, dim=1)
    kl = global_probs * (torch.log(global_probs + 1e-10) - local_log_probs)
    return kl[:, vc_idx, :, :].sum(dim=1).mean()


def logit_suppression(logits, targets, class_freq):
    _, classes, _, _ = logits.shape
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, classes)
    targets_flat = targets.reshape(-1)
    total_loss = torch.tensor(0.0, device=logits.device)
    for class_index in range(classes):
        mask = (targets_flat != class_index).float()
        if mask.sum() == 0:
            continue
        column = logits_flat[:, class_index]
        maximum = column.max().detach()
        mean_exp = (mask * torch.exp(column - maximum)).sum() / mask.sum()
        total_loss = total_loss + class_freq[class_index] * (torch.log(mean_exp + 1e-10) + maximum)
    return total_loss


class FedVLSClientTrainer(ClientTrainer):
    """FedVLS objective with fixed vacant-class distillation."""

    def __init__(self, *args, class_freq, vacant_mask, lam=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_freq = torch.as_tensor(class_freq, dtype=torch.float)
        self.vacant_mask = torch.as_tensor(vacant_mask, dtype=torch.bool)
        self.lam = lam

    def train(self, global_state=None):
        if global_state is None:
            global_state = deepcopy(self.model.state_dict())
        global_model = deepcopy(self.model).to(self.device)
        global_model.load_state_dict(global_state)
        global_model.eval()
        class_freq = self.class_freq.to(self.device)
        vacant_mask = self.vacant_mask.to(self.device)
        from fedseismic.losses import DiceLoss
        dice = DiceLoss()
        self.model.train()
        self.model.to(self.device)
        self.optimizer = self._make_optimizer()
        for _ in range(self.local_epochs):
            for images, targets, _ in self.loader:
                images = images.to(self.device, dtype=torch.float)
                targets = targets.to(self.device, dtype=torch.long)
                logits = _logits(self.model(images))
                with torch.no_grad():
                    global_logits = _logits(global_model(images))
                loss = (
                    calibrated_cross_entropy(logits, targets, class_freq)
                    + dice(logits, targets)
                    + self.lam * vacant_class_distillation(logits, global_logits, vacant_mask)
                    + logit_suppression(logits, targets, class_freq)
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self


def class_frequency(loader, num_classes):
    dataset = getattr(loader, "dataset", None)
    labels = _dataset_labels(dataset)
    if labels is not None:
        values, counts = np.unique(labels, return_counts=True)
        frequencies = np.zeros(num_classes, dtype=np.float64)
        total = max(int(counts.sum()), 1)
        for value, count in zip(values, counts):
            frequencies[int(value)] = count / total
        return frequencies
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, targets, _ in loader:
        values, value_counts = np.unique(np.asarray(targets), return_counts=True)
        counts[values.astype(int)] += value_counts
    return counts / max(counts.sum(), 1.0)


def _dataset_labels(dataset):
    if dataset is None:
        return None
    if hasattr(dataset, "label") and hasattr(dataset, "indices"):
        return np.asarray(dataset.label)[:, dataset.indices, :].reshape(-1)
    if hasattr(dataset, "targets") and hasattr(dataset, "indices"):
        return np.asarray(dataset.targets)[list(dataset.indices)].reshape(-1)
    return None


class FeatureExtractor:
    """Small hook helper used by prototype-based FedSeis training."""

    def __init__(self, model, layer_name):
        self.features = None
        self.hook = getattr(model, layer_name).register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.features = output

    def remove(self):
        self.hook.remove()


def prototype_alignment_loss(features, targets, prototypes, valid_mask):
    _, dimensions, feature_height, feature_width = features.shape
    _, height, width = targets.shape
    if (feature_height, feature_width) != (height, width):
        targets = F.interpolate(
            targets.unsqueeze(1).float(), size=(feature_height, feature_width), mode="nearest",
        ).squeeze(1).long()
    features_flat = features.permute(0, 2, 3, 1).reshape(-1, dimensions)
    targets_flat = targets.reshape(-1)
    features_normalized = F.normalize(features_flat, dim=1)
    prototypes_normalized = F.normalize(prototypes, dim=1)
    selected = valid_mask[targets_flat]
    if selected.sum() == 0:
        return torch.tensor(0.0, device=features.device)
    cosine = (features_normalized[selected]
              * prototypes_normalized[targets_flat[selected]]).sum(dim=1)
    return (1.0 - cosine).mean()


class FedSeisClientTrainer(ClientTrainer):
    """Prototype alignment plus vacant-class distillation for FedSeis."""

    def __init__(self, *args, prototypes, proto_valid, vacant_mask, alpha_proto=0.5,
                 alpha_dis=0.1, proto_layer="up4", **kwargs):
        super().__init__(*args, **kwargs)
        self.prototypes = prototypes
        self.proto_valid = proto_valid
        self.vacant_mask = vacant_mask
        self.alpha_proto = alpha_proto
        self.alpha_dis = alpha_dis
        self.proto_layer = proto_layer

    def train(self, global_state=None):
        if global_state is None:
            global_state = deepcopy(self.model.state_dict())
        global_model = deepcopy(self.model).to(self.device)
        global_model.load_state_dict(global_state)
        global_model.eval()
        from fedseismic.losses import FocalDiceLoss
        extractor = FeatureExtractor(self.model, self.proto_layer)
        focal_dice = FocalDiceLoss()
        prototypes = self.prototypes.to(self.device)
        proto_valid = self.proto_valid.to(self.device)
        vacant_mask = self.vacant_mask.to(self.device)
        self.model.train()
        self.model.to(self.device)
        self.optimizer = self._make_optimizer()
        for _ in range(self.local_epochs):
            for images, targets, _ in self.loader:
                images = images.to(self.device, dtype=torch.float)
                targets = targets.to(self.device, dtype=torch.long)
                logits = _logits(self.model(images))
                l_proto = prototype_alignment_loss(
                    extractor.features, targets, prototypes, proto_valid,
                )
                if self.alpha_dis > 0 and vacant_mask.any():
                    with torch.no_grad():
                        global_logits = _logits(global_model(images))
                    l_dis = vacant_class_distillation(logits, global_logits, vacant_mask)
                else:
                    l_dis = torch.tensor(0.0, device=self.device)
                loss = focal_dice(logits, targets) + self.alpha_proto * l_proto + self.alpha_dis * l_dis
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        extractor.remove()
        return self
