"""Ch6: personalization gain per site (FedAvg deployed minus shared, present-class
mIoU on the site's own holdout), all F3 jobs with batch and group norm, seeds
0, 1, 42. Also FedAvg shared model with BN recomputed on the site's own
training data. CPU. Writes figures/ch6/data/f3_site_gain.csv."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd, torch
from fedseismic.config import RunConfig
from fedseismic.experiment import build_federated_run
from fedseismic.models import UNet
from fedseismic.privacy.f3_per_method import _present_miou, _recalibrate_bn
from fedseismic.privacy.log import load_global_state, load_local_states, load_meta

torch.set_num_threads(16)
rows = []
for norm in ("batch", "group"):
    for n in (3, 5):
        for seed in (0, 1, 42):
            run = ROOT / f"results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n{n}_{norm}norm/seed_{seed}"
            cfg = RunConfig.from_mapping(dict(load_meta(run)["config"], device="cpu"))
            data = build_federated_run(cfg, seed)
            def model(state):
                m = UNet(1, cfg.num_classes, bilinear=False, norm=cfg.norm, groups=cfg.norm_groups)
                m.load_state_dict(state); return m.eval()
            g = load_global_state(run); local = load_local_states(run)
            for site in range(n):
                hold = data.client_tests[site]
                row = dict(norm=norm, n=n, seed=seed, site=site,
                           deployed=_present_miou(model(local[site]), hold, 6, "cpu"),
                           shared=_present_miou(model(g), hold, 6, "cpu"))
                if norm == "batch":
                    m = model(g); _recalibrate_bn(m, data.loaders[site])
                    row["shared_recal"] = _present_miou(m, hold, 6, "cpu")
                rows.append(row); print(row, flush=True)
f = pd.DataFrame(rows); f["gain"] = f.deployed - f.shared
out = ROOT / "figures/ch6/data/f3_site_gain.csv"; f.to_csv(out, index=False)
print(f.groupby(["norm", "n", "site"])[["deployed", "shared", "gain"]].mean().round(3).to_string())
print(f.groupby(["norm", "n"])["gain"].agg(["mean", "min", "max", lambda x: (x > 0).mean()]).round(3).to_string())
