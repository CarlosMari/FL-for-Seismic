"""Disk layout for server-visible artifacts vs ground-truth histograms."""

import json
from pathlib import Path

import numpy as np
import torch

from .estimators import (
    batchnorm_tensors,
    extract_last_linear,
    flatten_state_delta,
    project_vector,
)

LABELS_NAME = "labels/pi.json"
META_NAME = "run_meta.json"
VIEW_DIR = "view"
LOCAL_DIR = "local"
GLOBAL_NAME = "global_final.pt"


def _as_path(run_dir):
    return Path(run_dir)


def labels_path(run_dir):
    return _as_path(run_dir) / LABELS_NAME


def meta_path(run_dir):
    return _as_path(run_dir) / META_NAME


def view_dir(run_dir):
    return _as_path(run_dir) / VIEW_DIR


def local_dir(run_dir):
    return _as_path(run_dir) / LOCAL_DIR


class PrivacyLogger:
    """Writes the honest-but-curious view and keeps π in a separate labels file."""

    def __init__(self, run_dir, projection_dim=4096, projection_seed=0, log_last_only=False,
                 save_local=True):
        self.run_dir = _as_path(run_dir)
        self.projection_dim = projection_dim
        self.projection_seed = int(projection_seed)
        self.log_last_only = bool(log_last_only)
        self.save_local = bool(save_local)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        view_dir(self.run_dir).mkdir(parents=True, exist_ok=True)
        labels_path(self.run_dir).parent.mkdir(parents=True, exist_ok=True)
        if self.save_local:
            local_dir(self.run_dir).mkdir(parents=True, exist_ok=True)

    def write_meta(self, payload):
        meta_path(self.run_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_ground_truth(self, histograms, rare_classes=()):
        """Store true π. Attack loaders must not treat this as server-visible."""
        body = {
            "rare_classes": list(rare_classes),
            "clients": [
                {"client": int(index), "pi": np.asarray(hist, dtype=np.float64).tolist()}
                for index, hist in enumerate(histograms)
            ],
        }
        labels_path(self.run_dir).write_text(json.dumps(body, indent=2), encoding="utf-8")

    def log_upload(self, round_index, client, uploaded, global_state, scalars=None,
                   is_last_round=False):
        if self.log_last_only and not is_last_round:
            if self.save_local:
                self.write_local(client, uploaded)
            return
        delta = flatten_state_delta(uploaded, global_state)
        projected = project_vector(delta, self.projection_dim, self.projection_seed)
        weight, bias = extract_last_linear(uploaded)
        payload = {
            "round": np.int32(round_index),
            "client": np.int32(client),
            "delta_proj": projected,
        }
        if weight is not None:
            payload["last_weight"] = np.asarray(weight, dtype=np.float32)
        if bias is not None:
            payload["last_bias"] = np.asarray(bias, dtype=np.float32)
        bn = batchnorm_tensors(uploaded)
        if bn:
            payload["bn_keys"] = np.array(list(bn.keys()))
            for key, value in bn.items():
                payload[f"bn::{key}"] = value
        path = view_dir(self.run_dir) / f"round_{int(round_index):03d}_client_{int(client):03d}.npz"
        np.savez_compressed(path, **payload)
        if scalars:
            scalar_path = path.with_suffix(".json")
            scalar_path.write_text(json.dumps(scalars, indent=2), encoding="utf-8")
        if self.save_local:
            self.write_local(client, uploaded)

    def write_local(self, client, state):
        torch.save(
            {key: tensor.detach().cpu() for key, tensor in state.items()},
            local_dir(self.run_dir) / f"client_{int(client):03d}.pt",
        )

    def write_global(self, state):
        torch.save(
            {key: tensor.detach().cpu() for key, tensor in state.items()},
            self.run_dir / GLOBAL_NAME,
        )


def load_labels(run_dir):
    payload = json.loads(labels_path(run_dir).read_text(encoding="utf-8"))
    histograms = {}
    for row in payload["clients"]:
        histograms[int(row["client"])] = np.asarray(row["pi"], dtype=np.float64)
    return histograms, tuple(payload.get("rare_classes") or ())


def load_meta(run_dir):
    path = meta_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_views(run_dir):
    """Server-visible artifacts only (no labels/pi.json)."""
    records = []
    for path in sorted(view_dir(run_dir).glob("round_*_client_*.npz")):
        with np.load(path, allow_pickle=False) as handle:
            record = {key: handle[key] for key in handle.files}
        record["round"] = int(record["round"])
        record["client"] = int(record["client"])
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            record["scalars"] = json.loads(sidecar.read_text(encoding="utf-8"))
        records.append(record)
    return records


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_local_states(run_dir):
    states = {}
    folder = local_dir(run_dir)
    if not folder.exists():
        return states
    for path in sorted(folder.glob("client_*.pt")):
        client = int(path.stem.split("_")[1])
        states[client] = _load_checkpoint(path)
    return states


def load_global_state(run_dir):
    path = _as_path(run_dir) / GLOBAL_NAME
    if not path.exists():
        return None
    return _load_checkpoint(path)


def latest_view_by_client(records):
    latest = {}
    for record in records:
        client = record["client"]
        if client not in latest or record["round"] > latest[client]["round"]:
            latest[client] = record
    return latest
