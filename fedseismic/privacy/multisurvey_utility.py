"""Per-survey rescore for multisurvey runs (plan v3 go/no-go, condition 1).

Scores each survey's own client holdouts with present-class mIoU, for the
deployed locals, the shared global, and (not a decision number) official-test
F3. Reports a per-survey personalization gain = local_mIoU - global_on_local.
No re-training: weights come from the saved checkpoints.
"""

from pathlib import Path
import json

import numpy as np
import torch


def _survey_of(client, per_survey, survey_solo=None):
    if survey_solo:
        return str(survey_solo).upper()
    return "A" if client < per_survey else "B"


def _score_deployed(model, loader, num_classes, class_block, device):
    """Present-class mIoU with argmax masked to ``class_block``.

    FedPer/FedPerBN deploy a per-survey model: only the client's own class
    rows are trained, the rest sit at init and would win an unmasked argmax.
    Masking is the deployed model's own semantics, not a trick.

    Returns ``(present_mean, per_class_iou, present_mask)`` — the per-class
    breakdown (and which classes actually occur at the site) is the "real
    mIoU" we record per site, not just the macro mean.
    """
    from fedseismic.eval.metrics import _logits, _present_mean
    from sklearn.metrics import jaccard_score

    model.eval()
    model.to(device)
    lo, hi = class_block
    mask = torch.zeros(num_classes, dtype=torch.bool)
    mask[lo:hi] = True
    preds, targs = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            logits = _logits(model(images.to(device, dtype=torch.float)))
            logits = logits.clone()
            logits[:, ~mask.to(device)] = -1e9
            preds.append(logits.argmax(dim=1).cpu().numpy().ravel())
            targs.append(np.asarray(labels.detach().cpu() if torch.is_tensor(labels) else labels).ravel())
    if not preds:
        nan = float("nan")
        return nan, np.full(num_classes, nan), np.zeros(num_classes, dtype=bool)
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targs)
    per = jaccard_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)
    present = np.asarray(
        [np.any(y_true == k) for k in range(num_classes)], dtype=bool
    )
    return _present_mean(per, y_true, num_classes), np.asarray(per), present


def rescore_run(run_dir, device="cuda"):
    from fedseismic.config import RunConfig, resolve_device
    from fedseismic.experiment import build_federated_run
    from fedseismic.eval.metrics import score_loader
    from fedseismic.models import UNet
    from fedseismic.privacy.log import (
        load_global_state, load_local_states, load_meta, load_personal_states,
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

    # FedPer/FedPerBN/FedPerSurvey deploy the personal model (body + own head +
    # own BN); local/ holds the upload, which is the wrong head for evaluation.
    algo = str(cfg.algorithm)
    uses_personal = algo in ("fedper", "fedperbn", "fedpersurvey")
    deployed_states = (
        load_personal_states(run_dir) if uses_personal else load_local_states(run_dir)
    )

    def _survey_block(client):
        survey = _survey_of(client, per_survey, cfg.survey_solo)
        return (0, 6) if survey == "A" else (6, 12)

    locals_by_survey = {"A": [], "B": []}
    per_class_by_survey = {"A": [], "B": []}
    per_site_rows = []
    for client, state in sorted(deployed_states.items()):
        if client >= len(data.client_tests):
            continue
        loader = data.client_tests[client]
        if loader is None or not len(loader.dataset):
            continue
        model = _new()
        model.load_state_dict(state)
        if uses_personal:
            score, per_class, present = _score_deployed(
                model, loader, cfg.num_classes, _survey_block(client), device
            )
        else:
            score, per_class, _ = score_loader(
                model, loader, cfg.num_classes, device, cfg.task
            )
            per_class = np.asarray(per_class, dtype=np.float64)
            present = np.asarray(
                [np.any(np.concatenate([
                    np.asarray(t.detach().cpu() if torch.is_tensor(t) else t).ravel()
                    for _, t, _ in loader
                ]) == k) for k in range(cfg.num_classes)], dtype=bool
            )
        survey = _survey_of(client, per_survey, cfg.survey_solo)
        locals_by_survey[survey].append(score)
        per_class_by_survey[survey].append(per_class)
        for k in range(cfg.num_classes):
            per_site_rows.append({
                "survey": survey, "client": client, "class": k,
                "iou": float(per_class[k]) if k < len(per_class) else float("nan"),
                "present": bool(present[k]) if k < len(present) else False,
            })

    global_state = load_global_state(run_dir)
    globals_by_survey = {"A": [], "B": []}
    # FedBN/FedPerBN have no shared normalisation or shared head (shared ≡
    # deployed per the Chapter 4 convention), so their "global" is not a
    # defined reference. Skip it.
    is_local_only = str(cfg.algorithm) in ("fedbn", "fedperbn", "fedpersurvey")
    if global_state is not None and not is_local_only:
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
        # Real per-class mIoU: mean over the survey's sites, one value per class.
        if per_class_by_survey[name]:
            stacked = np.vstack(per_class_by_survey[name])
            out[f"{name}_local_per_class_iou"] = np.nanmean(stacked, axis=0).tolist()
        else:
            out[f"{name}_local_per_class_iou"] = []
    out["_per_site_class_rows"] = per_site_rows
    return out


def main(roots=("results/runs/multisurvey", "results/runs/multisurvey_fedbn",
                "results/runs/multisurvey_fedperbn", "results/runs/multisurvey_fedpersurvey",
                "results/runs/multisurvey_fedlc",
                "results/runs/multisurvey_solo_a", "results/runs/multisurvey_solo_b")):
    import pandas as pd

    rows = []
    detail_rows = []
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
            detail = scored.pop("_per_site_class_rows", [])
            for r in detail:
                r.update({
                    "run": str(run), "algorithm": cfg.get("algorithm"),
                    "seed": meta.get("seed"), "survey_solo": cfg.get("survey_solo"),
                })
            detail_rows.extend(detail)
            rows.append({
                "run": str(run),
                "algorithm": cfg.get("algorithm"),
                "logit_adjust": cfg.get("logit_adjust"),
                "survey_solo": cfg.get("survey_solo"),
                **scored,
            })
            flat = {k: (round(v, 3) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                    for k, v in scored.items()}
            print("  ", flat, flush=True)
    frame = pd.DataFrame(rows)
    out = Path("results/tables/multisurvey_utility.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out.resolve()}")
    if detail_rows:
        det = pd.DataFrame(detail_rows)
        det_out = Path("results/tables/multisurvey_per_class.csv")
        det.to_csv(det_out, index=False)
        print(f"wrote {det_out.resolve()} ({len(det)} per-site class rows)")
    return frame


if __name__ == "__main__":
    main()
