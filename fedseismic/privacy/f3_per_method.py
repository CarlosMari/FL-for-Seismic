"""Per-method F3 table with the FedBN shared-model convention resolved.

Convention (advisor ruling, pending Carlos sign-off):
- FedAvg deployed model = global weights + averaged BN (what FedAvg aggregates).
- FedBN deployed model = shared weights + each site's own BN (shared is
  therefore N/A; there is no separate global model).
- Official-test for FedBN = each client's deployed model on the global test
  set, averaged over clients. Never a synthetic averaged-BN model.
- Sanity check: FedAvg weights + BN recalibrated on each site's train data.

All scores present-class mIoU (absent facies excluded from the mean).
"""

from pathlib import Path
import json

import numpy as np
import torch


def _present_miou(model, loader, num_classes, device):
    from fedseismic.eval.metrics import score_loader

    return score_loader(model, loader, num_classes, device, "segmentation")[0]


def _recalibrate_bn(model, loader, maxb=30):
    """Recompute BN running stats on this loader's data (train mode forward)."""
    model.train()
    n = 0
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(next(model.parameters()).device, dtype=torch.float)
            model(images)
            n += 1
            if n >= maxb:
                break
    model.eval()


def score_job(run_dir, device="cuda"):
    from fedseismic.config import RunConfig, resolve_device
    from fedseismic.experiment import build_federated_run
    from fedseismic.eval.metrics import score_loader
    from fedseismic.models import UNet
    from fedseismic.privacy.log import load_global_state, load_local_states, load_meta

    device = resolve_device(device)
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    seed = int(meta.get("seed", cfg.seed))
    data = build_federated_run(cfg, seed)

    def _new():
        return UNet(1, cfg.num_classes, bilinear=False, norm=cfg.norm, groups=cfg.norm_groups)

    official = data.tests.get("test1")
    official_loader = official[0] if isinstance(official, tuple) else official

    # Local (deployed -> own site test)
    locals_ = []
    for client, state in sorted(load_local_states(run_dir).items()):
        if client >= len(data.client_tests):
            continue
        loader = data.client_tests[client]
        if loader is None or not len(loader.dataset):
            continue
        m = _new(); m.load_state_dict(state)
        locals_.append(_present_miou(m, loader, cfg.num_classes, device))

    gstate = load_global_state(run_dir)
    is_fedbn = str(cfg.algorithm) == "fedbn"
    out = {
        "job": run_dir.parent.name,
        "seed": run_dir.name,
        "algorithm": cfg.algorithm,
        "norm": cfg.norm,
        "n_sites": cfg.num_clients,
        "local": float(np.mean(locals_)) if locals_ else float("nan"),
    }

    if is_fedbn:
        # shared ≡ deployed; no separate shared baseline.
        out["shared_on_local"] = float("nan")
        out["shared_is_na"] = True
        # official = each client's deployed model (own BN) on official test.
        deployed_official = []
        for client, state in sorted(load_local_states(run_dir).items()):
            if client >= len(data.client_tests):
                continue
            m = _new(); m.load_state_dict(state)
            if official_loader is not None:
                deployed_official.append(_present_miou(m, official_loader, cfg.num_classes, device))
        out["official"] = float(np.mean(deployed_official)) if deployed_official else float("nan")
    else:
        out["shared_is_na"] = False
        if gstate is not None:
            m = _new(); m.load_state_dict(gstate)
            shared = []
            for loader in data.client_tests:
                if loader is not None and len(loader.dataset):
                    shared.append(_present_miou(m, loader, cfg.num_classes, device))
            out["shared_on_local"] = float(np.mean(shared)) if shared else float("nan")
            if official_loader is not None:
                out["official"] = _present_miou(m, official_loader, cfg.num_classes, device)

    # Sanity check: FedAvg weights + BN recalibrated on each site's train data.
    if gstate is not None and str(cfg.norm).startswith("batch"):
        m = _new(); m.load_state_dict(gstate)
        rec = []
        for client, loader in enumerate(data.loaders):
            if client >= len(data.client_tests):
                continue
            tloader = data.client_tests[client]
            if loader is None or tloader is None or not len(tloader.dataset):
                continue
            _recalibrate_bn(m, loader)
            rec.append(_present_miou(m, tloader, cfg.num_classes, device))
        out["fedavg_localbn_recal"] = float(np.mean(rec)) if rec else float("nan")
    return out


def main(root="results/runs/seismic_f3", device="cuda"):
    import pandas as pd

    rows = []
    for run in sorted(Path(root).glob("*/seed_*")):
        if not (run / "results.json").is_file():
            continue
        print(f"score {run}", flush=True)
        try:
            rows.append(score_job(run, device))
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
    frame = pd.DataFrame(rows)
    out = Path("results/tables/f3_per_method.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out.resolve()}")
    return frame


if __name__ == "__main__":
    main()
