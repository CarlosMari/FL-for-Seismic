"""Ch6 audit: per-survey present-class mIoU of every pre-fix per-survey-head run
(head-aggregation bug, commit e8dcd9c fixed it on 30 Oct) and the matching solos,
seed 0, deployed personal states, CPU. Writes figures/ch6/data/old_multisurvey.csv."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch
from fedseismic.privacy.multisurvey_utility import rescore_run

RUNS = ["multisurvey_fedpersurvey", "multisurvey_fedpersurvey_buf50", "multisurvey_fedpersurvey_rot",
        "multisurvey_solo_k6_a", "multisurvey_solo_k6_b", "multisurvey_solo_k6_a_buf50",
        "multisurvey_solo_k6_b_buf50", "multisurvey_solo_k6_b_rot",
        "multisurvey3_fedpersurvey", "multisurvey3_fedpersurvey_rot", "multisurvey_solo_k6_c"]
NAMES = {"A": "F3", "B": "Parihaka", "C": "Penobscot"}
import os
DEVICE = os.environ.get("DEVICE", "cpu")
torch.set_num_threads(12)
rows = []
for name in RUNS:
    run = next((ROOT / "results/runs" / name).glob("*/seed_0"))
    s = rescore_run(run, device=DEVICE)
    for code in "ABC":
        v = s.get(f"{code}_local_miou", np.nan)
        if v == v:
            rows.append(dict(run=name, survey=NAMES[code], miou_present=float(v)))
            print(rows[-1], flush=True)
out = ROOT / "figures/ch6/data/old_multisurvey.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(pd.DataFrame(rows).pivot(index="run", columns="survey", values="miou_present").round(3).to_string())
