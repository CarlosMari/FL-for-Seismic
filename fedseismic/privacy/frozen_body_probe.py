"""Frozen-body probe: are the FedBN body features good enough for solo-level
accuracy, if we only retrain a fresh per-survey head?

Freeze the FedBN multi-survey body (shared weights + each site's own BN),
replace the 12-class outc with a fresh 6-class head, train ONLY the head on
that site's train data, and score its holdout. Compare to solo (~0.80) and
multi (~0.68/0.77).

- Probe ≈ solo (0.80): body features are fine; the gap was loss/head training.
- Probe ≈ multi (0.68/0.77): body features are degraded by sharing.

Report per-client to match the client-1 heterogeneity flag.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


def _class_block(survey):
    return (0, 6) if str(survey).upper() == "A" else (6, 12)


def probe_run(run_dir, epochs=5, lr=1e-2, device="cpu"):
    from fedseismic.config import RunConfig
    from fedseismic.eval.metrics import score_loader
    from fedseismic.experiment import build_federated_run
    from fedseismic.models import UNet
    from fedseismic.privacy.log import load_local_states, load_meta

    device = torch.device(device)
    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    cfg = RunConfig.from_mapping(meta["config"])
    seed = int(meta.get("seed", cfg.seed))
    data = build_federated_run(cfg, seed)
    states = load_local_states(run_dir)
    per_survey = cfg.num_clients // 2 if not cfg.survey_solo else cfg.num_clients

    results = []
    for client, state in sorted(states.items()):
        loader = data.loaders[client]
        tloader = data.client_tests[client]
        if loader is None or tloader is None or not len(tloader.dataset):
            continue
        survey = (str(cfg.survey_solo).upper() if cfg.survey_solo
                  else ("A" if client < per_survey else "B"))
        lo, hi = _class_block(survey)
        n_out = hi - lo

        model = UNet(1, cfg.num_classes, bilinear=False, norm=cfg.norm, groups=cfg.norm_groups)
        model.load_state_dict(state)
        model.to(device)
        # Freeze everything except the final classifier.
        for p in model.parameters():
            p.requires_grad = False
        model.outc.conv = nn.Conv2d(64, n_out, kernel_size=1).to(device)
        for p in model.outc.conv.parameters():
            p.requires_grad = True

        opt = torch.optim.AdamW(model.outc.conv.parameters(), lr=lr, weight_decay=1e-4)
        model.train()
        for _ in range(epochs):
            for images, labels, _ in loader:
                labels = labels.to(device) - lo  # map to 0..n_out-1
                logits = model(images.to(device, dtype=torch.float))
                logits = logits if isinstance(logits, torch.Tensor) else logits[0]
                loss = nn.functional.cross_entropy(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()

        # Score holdout with the fresh head, present-class mIoU over the block.
        from sklearn.metrics import jaccard_score
        preds, targs = [], []
        with torch.no_grad():
            for images, labels, _ in tloader:
                logits = model(images.to(device, dtype=torch.float))
                logits = logits if isinstance(logits, torch.Tensor) else logits[0]
                preds.append(logits.argmax(1).cpu().numpy().ravel())
                targs.append((labels.cpu().numpy().ravel() - lo))
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(targs)
        per = jaccard_score(y_true, y_pred, labels=list(range(n_out)), average=None, zero_division=0)
        present = np.bincount(y_true, minlength=n_out) > 0
        miou = float(np.mean(per[present])) if present.any() else float("nan")
        results.append({"survey": survey, "client": client, "probe_miou": miou,
                        "per_class": np.round(per, 3).tolist()})
        print(f"survey {survey} client {client} frozen-body-probe miou={miou:.4f} per-class={np.round(per,3)}", flush=True)
    return results


def main(run="results/runs/multisurvey_fedbn/fedbn_noniid_a0.1_lNA_dna_n4/seed_0", device="cpu"):
    probe_run(run, device=device)


if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:2] or ["results/runs/multisurvey_fedbn/fedbn_noniid_a0.1_lNA_dna_n4/seed_0"]),
         **({"device": sys.argv[2]} if len(sys.argv) > 2 else {}))
