"""Recover last-round outc-head deltas for the seismic grid.

The grid ran with a version of ``extract_last_linear`` that matched the
auxiliary ``reconstruct`` head instead of ``outc`` (UNet keys are
``outc.conv.weight``, not ``outc.weight``), so ``head/*.npz`` holds
reconstruction deltas. Per-round outc deltas are unrecoverable (only the
final upload per client was saved), but the last round is: ``local/`` holds
each client's final upload and ``received/`` holds every round's starting
model. This writes ``head_outc/round_020_client_*.npz`` without touching
``head/``.
"""

from pathlib import Path

import numpy as np

from fedseismic.privacy.estimators import extract_last_linear
from fedseismic.privacy.log import _load_checkpoint


def recover_run(run_dir, last_round=20):
    run_dir = Path(run_dir)
    received = _load_checkpoint(run_dir / "received" / f"round_{last_round:03d}.pt")
    received_weight, received_bias = extract_last_linear(received)
    out = Path(run_dir) / "head_outc"
    out.mkdir(parents=True, exist_ok=True)
    recovered = 0
    for path in sorted((run_dir / "local").glob("client_*.pt")):
        client = int(path.stem.split("_")[1])
        uploaded = _load_checkpoint(path)
        local_weight, local_bias = extract_last_linear(uploaded)
        if local_weight is None:
            continue
        weight_delta = np.asarray(local_weight, dtype=np.float64) - np.asarray(received_weight, dtype=np.float64)
        bias_delta = np.asarray(local_bias, dtype=np.float64).reshape(-1) - np.asarray(received_bias, dtype=np.float64).reshape(-1)
        np.savez_compressed(
            out / f"round_{last_round:03d}_client_{client:03d}.npz",
            weight_delta=weight_delta.astype(np.float32),
            bias_delta=bias_delta.astype(np.float32).reshape(-1),
        )
        recovered += 1
    return recovered


def main():
    root = Path("results/runs/seismic_f3")
    total = 0
    for job in sorted(root.iterdir()):
        if not job.is_dir():
            continue
        for run in sorted(job.glob("seed_*")):
            if not (run / "results.json").is_file():
                continue
            n = recover_run(run)
            total += n
            print(f"{job.name}/{run.name}: {n} clients", flush=True)
    print(f"recovered {total} last-round outc deltas")


if __name__ == "__main__":
    main()
