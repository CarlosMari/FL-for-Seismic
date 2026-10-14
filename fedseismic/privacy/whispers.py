"""Label-distribution inference from per-class accuracy over time.

Ma, Gao, Huang, and Wang, arXiv:2504.21436. The server scores each upload on a
public probe and stacks those per-class accuracies into a rounds-by-classes
matrix. An LSTM with temporal attention, trained on virtual clients whose
mix is known and whose set size matches the victim, maps that matrix to a
class mix. The victim's set size is given, so a miss is the inference model.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fedseismic.privacy.metrics import leakage_from_tv, to_simplex, total_variation


def class_accuracy(model, loader, num_classes, device, max_batches=None):
    """Per-class accuracy of ``model`` on a public loader. Missing classes are 0."""
    model.eval()
    model.to(device)
    correct = np.zeros(num_classes, dtype=np.float64)
    total = np.zeros(num_classes, dtype=np.float64)
    seen = 0
    with torch.no_grad():
        for images, targets, *_ in loader:
            images = images.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long).view(-1)
            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.ndim == 4:
                logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
                targets = targets.view(-1)
            predicted = logits.argmax(dim=1).view(-1)
            for class_index in range(num_classes):
                mask = targets == class_index
                count = int(mask.sum().item())
                total[class_index] += count
                if count:
                    correct[class_index] += int((predicted[mask] == class_index).sum().item())
            seen += 1
            if max_batches is not None and seen >= max_batches:
                break
    return np.divide(correct, np.maximum(total, 1.0))


class TemporalInference(nn.Module):
    """LSTM over the rounds-by-classes accuracy matrix, then temporal attention."""

    def __init__(self, num_classes, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(num_classes, hidden, batch_first=True)
        self.score = nn.Linear(hidden, 1)
        self.readout = nn.Linear(hidden, num_classes)

    def forward(self, matrices, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            matrices, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        output, _ = self.lstm(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        weights = self.score(hidden).squeeze(-1)
        mask = torch.arange(hidden.shape[1], device=hidden.device)[None, :] >= lengths[:, None]
        weights = weights.masked_fill(mask, -1e9)
        attention = torch.softmax(weights, dim=1)
        context = torch.sum(hidden * attention[:, :, None], dim=1)
        return torch.softmax(self.readout(context), dim=-1)


def train_inference(matrices, mixes, epochs=80, hidden=32, lr=1e-2, seed=0):
    """Fit the LSTM. ``matrices`` are lists of per-round class-accuracy rows."""
    torch.manual_seed(seed)
    num_classes = len(mixes[0])
    device = torch.device("cpu")
    model = TemporalInference(num_classes, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    packed, lengths = _pack(matrices)
    target = torch.tensor(np.stack([to_simplex(row) for row in mixes]), dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        predicted = model(packed, lengths)
        loss = F.kl_div(predicted.clamp(min=1e-8).log(), target, reduction="batchmean")
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def predict_mixes(model, matrices):
    packed, lengths = _pack(matrices)
    with torch.no_grad():
        predicted = model(packed, lengths).cpu().numpy()
    return [to_simplex(row) for row in predicted]


def advantage_against_mean(truths, predicted, baseline):
    """Mean (1 − TV) of the attack minus the same score for one shared baseline mix."""
    attack = [
        leakage_from_tv(total_variation(truth, hat))
        for truth, hat in zip(truths, predicted)
    ]
    floor = [
        leakage_from_tv(total_variation(truth, baseline))
        for truth in truths
    ]
    return float(np.mean(attack) - np.mean(floor))


def _pack(matrices):
    lengths = [len(rows) for rows in matrices]
    width = matrices[0][0].shape[-1] if np.ndim(matrices[0][0]) else len(matrices[0][0])
    padded = np.zeros((len(matrices), max(lengths), width), dtype=np.float32)
    for index, rows in enumerate(matrices):
        stacked = np.stack([np.asarray(row, dtype=np.float32) for row in rows])
        padded[index, : len(rows)] = stacked
    return torch.tensor(padded), torch.tensor(lengths, dtype=torch.long)


def virtual_matrices(cfg, global_state, model_factory, images, labels, probe, client_size,
                     n_channels, n_virtual=32, steps=4, alpha=0.1, seed=0, device="cpu"):
    """Trajectories of virtual clients with a known mix and the victim's set size.

    Each client trains ``steps`` successive local runs from the same starting
    global weights, and the public per-class accuracy is recorded after each
    run. The true mix is the subsample of length ``client_size``.
    """
    from fedseismic.config import DatasetName
    from fedseismic.data import build_cifar_loaders, build_medmnist_loaders, partition_dirichlet
    from fedseismic.privacy.debiased_softmax import _trainer

    rng = np.random.RandomState(seed)
    partitions = partition_dirichlet(labels, n_virtual, alpha, rng)
    sized = []
    for indices in partitions:
        indices = np.asarray(indices)
        if len(indices) < 2:
            continue
        if len(indices) > client_size:
            indices = rng.choice(indices, size=client_size, replace=False)
        sized.append(indices)
    if cfg.dataset == DatasetName.CIFAR10.value:
        loaders = build_cifar_loaders(images, labels, sized, cfg.batch_size, train=True)
    else:
        loaders = build_medmnist_loaders(
            images, labels, sized, cfg.batch_size, n_channels, train=True,
        )
    matrices = []
    mixes = []
    for loader in loaders:
        if len(loader.dataset) < 2:
            continue
        taken = np.asarray(loader.dataset.targets)[list(loader.dataset.indices)]
        mixes.append(to_simplex(np.bincount(taken, minlength=cfg.num_classes).astype(np.float64)))
        trainer = _trainer(cfg, model_factory(), loader, device)
        trainer.num_classes = cfg.num_classes
        trainer.logit_adjust = bool(getattr(cfg, "logit_adjust", False))
        trainer.local_epochs = max(1, int(cfg.local_epochs))
        trainer.download(global_state)
        rows = []
        for _ in range(steps):
            trainer.train(global_state=global_state)
            rows.append(class_accuracy(trainer.model, probe, cfg.num_classes, device))
        matrices.append(rows)
    return matrices, mixes


def load_client_matrices(run_dir):
    """Per-client sequence of public per-class accuracies, in round order."""
    import json

    from fedseismic.privacy.log import load_labels

    run_dir = Path(run_dir)
    histograms, _ = load_labels(run_dir)
    by_client = {}
    with (run_dir / "round_scores.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "class_acc" not in record:
                continue
            client = int(record["client"])
            by_client.setdefault(client, []).append(
                (int(record["round"]), np.asarray(record["class_acc"], dtype=np.float64))
            )
    matrices = {}
    mixes = {}
    for client, series in by_client.items():
        series.sort(key=lambda item: item[0])
        # One row per round. A repeated round is the same participation written twice.
        by_round = {round_index: row for round_index, row in series}
        matrices[client] = [by_round[round_index] for round_index in sorted(by_round)]
        mixes[client] = to_simplex(histograms[client])
    return matrices, mixes
