"""Atomic round-boundary checkpoints for the FedHCA2-style experiment."""

from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import platform
import random

import numpy as np
import scipy
import torch

from .fedhca2 import FedHCA2Aggregator


def source_fingerprint():
    """Fingerprint the Python package, independent of documentation commits."""
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def dataset_fingerprints(config):
    """Hash each configured input file once, not once per training round."""
    fields = ("train_seismic", "train_labels", "survey_b_seismic", "survey_b_labels",
              "survey_c_seismic", "survey_c_labels", "test1_seismic", "test1_labels",
              "test2_seismic", "test2_labels", "validation_seismic", "validation_labels")
    result = {}
    for field in fields:
        value = getattr(config, field, None)
        if value is None:
            continue
        digest = hashlib.sha256()
        with Path(value).open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        result[field] = digest.hexdigest()
    return result


def _rng_state(rng):
    name, keys, pos, has_gauss, cached = rng.get_state()
    return (name, keys.tolist(), pos, has_gauss, cached)


def _set_rng_state(rng, state):
    name, keys, pos, has_gauss, cached = state
    rng.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))


def split_fingerprints(server):
    """Hash the ordered section indices used by each train/test loader."""
    out = {}
    for kind, loaders in (("train", server.client_loaders), ("test", server.client_test_loaders)):
        for client, loader in enumerate(loaders):
            indices = getattr(loader.dataset, "indices", None)
            if indices is None:
                raise ValueError(f"{kind} client {client} has no explicit indices")
            encoded = np.asarray(indices, dtype="<i8").tobytes()
            out[f"{kind}:{client}"] = hashlib.sha256(encoded).hexdigest()
    return out


def _plain(value):
    """Keep checkpoint metadata compatible with torch's weights-only loader."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _signature(config):
    values = config.to_dict()
    for key in ("output_dir", "save_round_checkpoint", "resume_round_checkpoint"):
        values.pop(key, None)
    return values


def save_round_checkpoint(server, path, git_head=None):
    if server.config.algorithm != "fedhca2" or server._fedhca2 is None:
        raise ValueError("round checkpoints currently support FedHCA2 only")
    if len(server._client_trainers) != server.config.num_clients:
        raise ValueError("a FedHCA2 checkpoint needs every client state")
    checkpoint = {
        "format_version": 1, "git_head": git_head,
        "seed": server.run_seed,
        "source_fingerprint": server.source_fingerprint,
        "dataset_fingerprints": server.dataset_fingerprints,
        "split_fingerprints": split_fingerprints(server),
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
        "scipy_version": str(scipy.__version__), "python_version": platform.python_version(),
        "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "config": _plain(_signature(server.config)),
        "round": len(server.history),
        "history": [_plain(asdict(record)) for record in server.history],
        "global": {k: v.detach().cpu() for k, v in server.model.state_dict().items()},
        "personal": {i: trainer.personal_state for i, trainer in server._client_trainers.items()},
        "aggregator": server._fedhca2.checkpoint_state(),
        "server_rng": _rng_state(server.rng),
        "numpy_rng": _rng_state(np.random),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "logs": {
            name: (server.privacy_logger.run_dir / name).read_text()
            for name in ("history.jsonl", "round_scores.jsonl", "hyper_weights.jsonl")
            if (server.privacy_logger.run_dir / name).is_file()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_round_checkpoint(server, path, git_head=None):
    # Only checkpoints created by this program are supported; weights_only avoids
    # executing Python objects embedded in an untrusted file.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint["config"] != _plain(_signature(server.config)):
        raise ValueError("round checkpoint configuration does not match this run")
    if checkpoint.get("seed") != server.run_seed:
        raise ValueError("round checkpoint seed does not match this run")
    if (checkpoint.get("source_fingerprint") != server.source_fingerprint or
            checkpoint.get("dataset_fingerprints") != server.dataset_fingerprints or
            checkpoint.get("split_fingerprints") != split_fingerprints(server) or
            checkpoint.get("torch_version") != str(torch.__version__) or
            checkpoint.get("numpy_version") != str(np.__version__) or
            checkpoint.get("scipy_version") != str(scipy.__version__) or
            checkpoint.get("python_version") != platform.python_version() or
            checkpoint.get("cuda_version") != torch.version.cuda or
            checkpoint.get("cudnn_version") != torch.backends.cudnn.version()):
        raise ValueError("round checkpoint source, data, split or library versions do not match")
    if Path(path).resolve().parent == server.privacy_logger.run_dir.resolve():
        raise ValueError("resume into a new output directory to avoid stale partial logs")
    for name in checkpoint["logs"]:
        if (server.privacy_logger.run_dir / name).exists():
            raise ValueError("resume output directory already has round logs")
    from .server import RoundRecord

    server.model.load_state_dict(checkpoint["global"])
    server._fedhca2 = FedHCA2Aggregator.from_config(checkpoint["global"], server.config)
    server._fedhca2.restore_checkpoint_state(checkpoint["aggregator"])
    for client, state in checkpoint["personal"].items():
        server._trainer(int(client)).personal_state = state
        server.privacy_logger.write_personal(int(client), state)
    server.history = [RoundRecord(**record) for record in checkpoint["history"]]
    if checkpoint["round"] != len(server.history):
        raise ValueError("round checkpoint history length does not match")
    _set_rng_state(server.rng, checkpoint["server_rng"])
    _set_rng_state(np.random, checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    if checkpoint["cuda_rng"]:
        if not torch.cuda.is_available() or len(checkpoint["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("round checkpoint CUDA devices do not match")
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    for name, text in checkpoint["logs"].items():
        (server.privacy_logger.run_dir / name).write_text(text)
    return checkpoint["round"]
