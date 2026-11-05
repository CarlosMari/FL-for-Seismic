"""Fig 6.6 data: CPU rescore of the three-survey seed-0 arms, no training.

Writes figures/ch6/data/k3_{summary,per_class}.csv; results/tables is untouched.
The Parihaka solo stopped at 19/20 rounds with no results.json or run_meta.json,
so compare_multisurvey3_k3.py skips it. It is scored here with the solo A
run's config and survey_solo='b', from its round-19 personal state.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from fedseismic.config import RunConfig
from fedseismic.experiment import build_federated_run
from fedseismic.models import UNet
from fedseismic.privacy.log import load_personal_states
from fedseismic.privacy.multisurvey_utility import _score_deployed, rescore_run

R = ROOT / "results/runs"
ARMS = {
    "solo_F3": R / "multisurvey3_k3_solo_a/fedpersurvey_noniid_a0.1_lNA_dna_n1/seed_0",
    "solo_Penobscot": R / "multisurvey3_k3_solo_c/fedpersurvey_noniid_a0.1_lNA_dna_n1/seed_0",
    "fedpersurvey": R / "multisurvey3_k3_fedpersurvey/fedpersurvey_noniid_a0.1_lNA_dna_n3/seed_0",
    "fedhca2": R / "multisurvey3_k3_fedhca2_restart_s0/fedhca2_noniid_a0.1_lNA_dna_n3/seed_0",
}
SOLO_B = R / "multisurvey3_k3_solo_b/fedpersurvey_noniid_a0.1_lNA_dna_n1/seed_0"
BLOCKS = {"A": (0, 6), "B": (6, 12), "C": (12, 19)}
NAMES = {"A": "F3", "B": "Parihaka", "C": "Penobscot"}


def solo_b_scores():
    meta = json.loads((ARMS["solo_F3"] / "run_meta.json").read_text())
    config = dict(meta["config"], survey_solo="b")
    cfg = RunConfig.from_mapping(config)
    data = build_federated_run(cfg, int(meta.get("seed", cfg.seed)))
    model = UNet(1, cfg.num_classes, bilinear=False, norm=cfg.norm, groups=cfg.norm_groups)
    model.load_state_dict(load_personal_states(SOLO_B)[0])
    dev = os.environ.get("DEVICE", "cpu")
    model.to(dev)
    score, per, present = _score_deployed(model, data.client_tests[0], cfg.num_classes, BLOCKS["B"], dev)
    return float(score), np.asarray(per), np.asarray(present)


def main():
    torch.set_num_threads(20)
    summary, per_class = [], []

    def add(arm, code, score, per, present):
        lo, hi = BLOCKS[code]
        summary.append({"arm": arm, "survey": NAMES[code],
                        "miou_present": score,
                        "miou_own_all_class": float(np.mean(per[lo:hi])),
                        "miou_union19_all_class": float(np.sum(per[lo:hi]) / 19),
                        "n_present": int(np.sum(present[lo:hi]))})
        for k in range(lo, hi):
            per_class.append({"arm": arm, "survey": NAMES[code], "class": k - lo,
                              "iou": float(per[k]), "present": bool(present[k])})

    for arm, run in ARMS.items():
        print("rescore", arm, run, flush=True)
        scored = rescore_run(run, device=os.environ.get("DEVICE", "cpu"))
        rows = pd.DataFrame(scored["_per_site_class_rows"])
        for code in "ABC":
            if np.isnan(scored[f"{code}_local_miou"]):
                continue
            sub = rows[rows.survey == code].sort_values("class")
            add("solo" if arm.startswith("solo") else arm, code, scored[f"{code}_local_miou"],
                sub.iou.to_numpy(), sub.present.to_numpy())
    print("rescore solo_b", flush=True)
    add("solo", "B", *solo_b_scores())

    out = ROOT / "figures/ch6/data"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / "k3_summary.csv", index=False)
    pd.DataFrame(per_class).to_csv(out / "k3_per_class.csv", index=False)
    print(pd.DataFrame(summary).round(4).to_string())


if __name__ == "__main__":
    main()
