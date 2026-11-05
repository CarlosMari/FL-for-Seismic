"""Ch6: each F3 site trained alone against the same site in the federation.

For every N in {3, 5}, seed in {0, 1, 42}, and site, scores present-class mIoU
(score_loader, as fedseismic/privacy/f3_per_method.py) on
  - the site's own holdout, and
  - the official F3 test1 volume,
for: the solo model (results/runs/seismic_f3_site_solo), the FedAvg deployed
and shared models, and the FedBN deployed model (batch normalization).
CPU only. Writes figures/ch6/data/f3_site_solo.csv; results/ is untouched.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from fedseismic.config import RunConfig
from fedseismic.experiment import build_federated_run
from fedseismic.models import UNet
from fedseismic.privacy.f3_per_method import _present_miou
from fedseismic.privacy.log import load_global_state, load_local_states, load_meta

RUNS = ROOT / "results/runs"
OUT = ROOT / "figures/ch6/data/f3_site_solo.csv"


def main():
    torch.set_num_threads(16)
    rows = []
    for n in (3, 5):
        for seed in (0, 1, 42):
            fedavg = RUNS / f"seismic_f3/fedavg_noniid_a0.1_lNA_dna_n{n}_batchnorm/seed_{seed}"
            fedbn = RUNS / f"seismic_f3/fedbn_noniid_a0.1_lNA_dna_n{n}_batchnorm/seed_{seed}"
            cfg = RunConfig.from_mapping(dict(load_meta(fedavg)["config"], device="cpu"))
            data = build_federated_run(cfg, seed)
            test1 = data.tests["test1"]
            test1 = test1[0] if isinstance(test1, tuple) else test1

            def model(state):
                m = UNet(1, cfg.num_classes, bilinear=False, norm=cfg.norm, groups=cfg.norm_groups)
                m.load_state_dict(state)
                return m.eval()

            shared = model(load_global_state(fedavg))
            shared_test1 = _present_miou(shared, test1, cfg.num_classes, "cpu")
            avg_local, bn_local = load_local_states(fedavg), load_local_states(fedbn)
            for site in range(n):
                solo_dir = RUNS / f"seismic_f3_site_solo/fedavg_n{n}_site{site}_batch/seed_{seed}"
                if not (solo_dir / "results.json").is_file():
                    print("missing", solo_dir, flush=True)
                    continue
                holdout = data.client_tests[site]
                solo = model(load_global_state(solo_dir))
                deployed_bn = model(bn_local[site])
                row = {
                    "n": n, "seed": seed, "site": site,
                    "solo_holdout": _present_miou(solo, holdout, cfg.num_classes, "cpu"),
                    "solo_test1": _present_miou(solo, test1, cfg.num_classes, "cpu"),
                    "fedavg_deployed_holdout": _present_miou(model(avg_local[site]), holdout, cfg.num_classes, "cpu"),
                    "fedavg_shared_holdout": _present_miou(shared, holdout, cfg.num_classes, "cpu"),
                    "fedavg_shared_test1": shared_test1,
                    "fedbn_deployed_holdout": _present_miou(deployed_bn, holdout, cfg.num_classes, "cpu"),
                    "fedbn_deployed_test1": _present_miou(deployed_bn, test1, cfg.num_classes, "cpu"),
                }
                rows.append(row)
                print(row, flush=True)
    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(frame.groupby("n").mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
