"""FedHCA2-style aggregation for single-task, disjoint-label surveys.

Lu et al., CVPR 2024 (github.com/innovator-zero/FedHCA2), adapted. The paper
federates clients doing different tasks; here every client does segmentation
over its own disjoint facies, so one survey plays the role of one task:

- Encoder (``inc``, ``down*``): hyper conflict-averse aggregation.
- Decoder (``up*``): hyper cross-attention aggregation, layer by layer.
- Head (``outc``) and BatchNorm: local, never aggregated (disjoint label
  spaces + FedBN). The loss is masked to the survey's own rows.

Dropped because nothing here needs them: multi-task clients, per-task decoder
groups, task loss weights, and homo-client averaging (one client per survey).
Every client ends each round with its own encoder and decoder.
"""

import numpy as np
import torch
from scipy.optimize import minimize

from .client import ClientTrainer, _is_bn_key
from .survey_heads import FedPerSurveyBNClientTrainer

ENCODER_MODULES = ("inc", "down1", "down2", "down3", "down4")
DECODER_MODULES = ("up1", "up2", "up3", "up4")


def _module_keys(state_dict, modules):
    return [
        key for key in state_dict
        if key.split(".", 1)[0] in modules and not _is_bn_key(key)
    ]


def encoder_keys(state_dict):
    return _module_keys(state_dict, ENCODER_MODULES)


def decoder_keys(state_dict):
    return _module_keys(state_dict, DECODER_MODULES)


def get_ca_delta(flat_deltas, ca_c):
    """Conflict-averse update: the mean delta plus a push toward the worst client.

    Same optimum and scaling (``rescale=1``) as upstream ``get_ca_delta``. The
    Gram matrix is divided by its mean first so SLSQP's absolute tolerance does
    not depend on the size of the deltas.
    """
    grads = torch.stack(flat_deltas).t()  # (d, N)
    N = grads.shape[1]
    gram = grads.t().mm(grads).double().cpu().numpy()
    scale = float(gram.mean())
    if scale <= 0.0:
        return torch.zeros_like(grads[:, 0])
    A = gram / scale
    uniform = np.ones(N) / N

    def objective(x):
        return float(x @ A @ uniform + ca_c * np.sqrt(x @ A @ x + 1e-8))

    weights = minimize(
        objective, uniform, bounds=[(0.0, 1.0)] * N,
        constraints={"type": "eq", "fun": lambda x: 1.0 - x.sum()},
    ).x
    gw = (grads * torch.as_tensor(weights, dtype=grads.dtype).reshape(1, -1)).sum(1)
    lmbda = (ca_c * np.sqrt(scale) + 1e-8) / (float(gw.norm()) + 1e-8)
    return (grads.mean(1) + lmbda * gw) / (1.0 + ca_c ** 2)


def cross_attention_deltas(deltas):
    """Each client's delta attends over all clients' deltas. ``deltas``: (N, d)."""
    scores = deltas @ deltas.t() * deltas.shape[1] ** -0.5
    return torch.softmax(scores, dim=-1) @ deltas


def _flatten(state, keys):
    return torch.cat([state[key].reshape(-1).float() for key in keys])


def _inside_unit(values):
    """Where ``torch.clamp(values, 0, 1)`` has a non-zero derivative."""
    return ((values >= 0.0) & (values <= 1.0)).to(values.dtype)


class FedHCA2Aggregator:
    """Holds the hyperweights and turns trained states into per-client updates.

    Upstream backpropagates through saved outputs. Both updates are linear in
    their weight, so the gradients are written out directly and no graph over
    the model parameters is kept.
    """

    def __init__(self, initial_state, num_clients, ca_c=0.4, alpha_init=0.1,
                 beta_init=0.1, lr=0.01, momentum=0.9, weight_decay=1e-4):
        self.num_clients = int(num_clients)
        self.ca_c = float(ca_c)
        self.encoder_keys = encoder_keys(initial_state)
        self.decoder_keys = decoder_keys(initial_state)
        self.layers = sorted({key.rsplit(".", 1)[0] for key in self.decoder_keys})
        self.alpha = torch.nn.Parameter(torch.full((self.num_clients,), float(alpha_init)))
        self.beta = {
            layer: torch.nn.Parameter(torch.full((self.num_clients,), float(beta_init)))
            for layer in self.layers
        }
        self.optimizer = torch.optim.SGD(
            [self.alpha, *self.beta.values()], lr=lr, momentum=momentum,
            weight_decay=weight_decay,
        )
        shared = self.encoder_keys + self.decoder_keys
        self._last = [
            {key: initial_state[key].detach().cpu().float().clone() for key in shared}
            for _ in range(self.num_clients)
        ]
        self._encoder_update = None
        self._decoder_updates = None

    @classmethod
    def from_config(cls, initial_state, config):
        return cls(
            initial_state, config.num_clients, ca_c=config.fedhca2_ca_c,
            alpha_init=config.fedhca2_alpha_init, beta_init=config.fedhca2_beta_init,
            lr=config.fedhca2_hyper_lr, momentum=config.fedhca2_hyper_momentum,
            weight_decay=config.fedhca2_hyper_weight_decay,
        )

    def _update_hyperweights(self, trained):
        """Move the weights along this round's local training (upstream's rule)."""
        alpha_grad = torch.zeros_like(self.alpha)
        beta_grad = {layer: torch.zeros_like(self.beta[layer]) for layer in self.layers}
        for client in range(self.num_clients):
            last, state = self._last[client], trained[client]
            diff = _flatten(last, self.encoder_keys) - _flatten(state, self.encoder_keys)
            alpha_grad[client] = torch.dot(self._encoder_update, diff)
            for key in self.decoder_keys:
                layer = key.rsplit(".", 1)[0]
                beta_grad[layer][client] += torch.dot(
                    self._decoder_updates[client][key].reshape(-1),
                    (last[key] - state[key].float()).reshape(-1),
                )
        self.alpha.grad = alpha_grad * _inside_unit(self.alpha.detach())
        for layer in self.layers:
            self.beta[layer].grad = beta_grad[layer] * _inside_unit(self.beta[layer].detach())
        self.optimizer.step()

    def step(self, trained):
        """Return one ``{key: tensor}`` per client, in client order."""
        if len(trained) != self.num_clients:
            raise ValueError("FedHCA2 needs every client in every round")
        if self._encoder_update is not None:
            self._update_hyperweights(trained)
        trained = [
            {key: state[key].detach().cpu().float() for key in self._last[0]}
            for state in trained
        ]
        updated = [{} for _ in range(self.num_clients)]

        encoder_deltas = [
            _flatten(state, self.encoder_keys) - _flatten(last, self.encoder_keys)
            for state, last in zip(trained, self._last)
        ]
        self._encoder_update = get_ca_delta(encoder_deltas, self.ca_c)
        alpha = torch.clamp(self.alpha.detach(), 0.0, 1.0)
        for client in range(self.num_clients):
            start = 0
            for key in self.encoder_keys:
                count = self._last[client][key].numel()
                stop = start + count
                step = encoder_deltas[client][start:stop] + alpha[client] * self._encoder_update[start:stop]
                updated[client][key] = self._last[client][key] + step.reshape(self._last[client][key].shape)
                start = stop

        self._decoder_updates = [{} for _ in range(self.num_clients)]
        for key in self.decoder_keys:
            beta = torch.clamp(self.beta[key.rsplit(".", 1)[0]].detach(), 0.0, 1.0)
            shape = self._last[0][key].shape
            deltas = torch.stack([
                (trained[client][key] - self._last[client][key]).reshape(-1)
                for client in range(self.num_clients)
            ])
            attended = cross_attention_deltas(deltas)
            for client in range(self.num_clients):
                self._decoder_updates[client][key] = attended[client].reshape(shape)
                updated[client][key] = self._last[client][key] + (
                    deltas[client] + beta[client] * attended[client]
                ).reshape(shape)

        self._last = [{key: value.clone() for key, value in state.items()} for state in updated]
        return updated

    def weights(self):
        return {
            "alpha": self.alpha.detach().tolist(),
            "beta": {layer: self.beta[layer].detach().tolist() for layer in self.layers},
        }


class FedHCA2ClientTrainer(FedPerSurveyBNClientTrainer):
    """Masked loss like fedpersurvey, but the whole model is per client.

    The server's aggregate is only a placeholder here: a client resumes from
    its ``personal_state``, whose encoder and decoder ``receive`` overwrites
    after each aggregation. Head and BN are never touched by the server.
    """

    def _local_keys(self, state_dict):
        return {
            key for key in state_dict
            if _is_bn_key(key) or key.split(".", 1)[0] in ("outc", "reconstruct")
        }

    def train(self, global_state=None):
        super().train(global_state=global_state)
        self.model.load_state_dict(self.personal_state)
        return self

    def download(self, state_dict):
        ClientTrainer.download(self, state_dict)
        if self.personal_state is not None:
            self.model.load_state_dict(self.personal_state)
        return self

    def receive(self, shared):
        for key, value in shared.items():
            self.personal_state[key] = value.detach().clone().to(self.personal_state[key].dtype)
        return self
