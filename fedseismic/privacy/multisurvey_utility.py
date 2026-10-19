"""Per-survey rescore for multisurvey runs (plan v3 go/no-go, condition 1).

Scores each survey's own client holdouts with present-class mIoU, for the
deployed locals, the shared global, and (not a decision number) official-test
F3. Reports a per-survey personalization gain = local_mIoU - global_on_local.
No re-training: weights come from the saved checkpoints.
"""

from pathlib import Path
import json

import numpy as np


def _survey_of(client, per_survey, survey_solo=None):
    if survey_solo:
        return str(survey_solo).upper()
    return "A" if client < per_survey else "B"


def rescore_run(run_dir, device="cuda"):
    from fedseismic.config import RunConfig, resolve_device
    from fedseismic.experiment import build_federated_run
    from fedseismic.eval.metrics import score_loader
    from fedseismic.models import UNet
    from fedseismic.privacy.log import (
        load_global_state, load_local_states, load_meta,
    )

    device = resolve_device(device)
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    seed = int(meta.get("seed", cfg.seed))
    data = build_federated_run(cfg, seed)
    per_survey = cfg.num_clients // 2 if not cfg.survey_solo else cfg.num_clients

    def _new():
        return UNet(1, cfg.num_classes, bilinear=False, norm=cfg.norm,
                    groups=cfg.norm_groups)

    locals_by_survey = {"A": [], "B": []}
    for client, state in sorted(load_local_states(run_dir).items()):
        if client >= len(data.client_tests):
            continue
        loader = data.client_tests[client]
        if loader is None or not len(loader.dataset):
            continue
        model = _new()
        model.load_state_dict(state)
        score = score_loader(model, loader, cfg.num_classes, device, cfg.task)[0]
        locals_by_survey[_survey_of(client, per_survey, cfg.survey_solo)].append(score)

    global_state = load_global_state(run_dir)
    globals_by_survey = {"A": [], "B": []}
    if global_state is not None:
        for client, loader in enumerate(data.client_tests):
            if loader is None or not len(loader.dataset):
                continue
            model = _new()
            model.load_state_dict(global_state)
            score = score_loader(model, loader, cfg.num_classes, device, cfg.task)[0]
            globals_by_survey[_survey_of(client, per_survey, cfg.survey_solo)].append(score)

    out = {}
    for name in ("A", "B"):
        local = float(np.mean(locals_by_survey[name])) if locals_by_survey[name] else float("nan")
        glob = float(np.mean(globals_by_survey[name])) if globals_by_survey[name] else float("nan")
        out[f"{name}_local_miou"] = local
        out[f"{name}_global_on_local"] = glob
        out[f"{name}_gain"] = local - glob
    return out


def main(roots=("results/runs/multisurvey", "results/runs/multisurvey_fedlc",
                "results/runs/multisurvey_solo_a", "results/runs/multisurvey_solo_b")):
    import pandas as pd

    rows = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for run in sorted(root.glob("*/seed_*")):
            if not (run / "results.json").is_file():
                continue
            meta = json.loads((run / "run_meta.json").read_text())
            cfg = meta["config"]
            print(f"rescore {run}", flush=True)
            try:
                scored = rescore_run(run)
            except Exception as exc:
                print(f"  FAILED: {exc}", flush=True)
                continue
            rows.append({
                "run": str(run),
                "algorithm": cfg.get("algorithm"),
                "logit_adjust": cfg.get("logit_adjust"),
                "survey_solo": cfg.get("survey_solo"),
                **scored,
            })
            print("  ", {k: round(v, 3) for k, v in scored.items()}, flush=True)
    frame = pd.DataFrame(rows)
    out = Path("results/tables/multisurvey_utility.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out.resolve()}")
    return frame


if __name__ == "__main__":
    main()
