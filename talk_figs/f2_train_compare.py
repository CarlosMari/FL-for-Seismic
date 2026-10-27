"""Train the two F2 comparison models (whole-F3 + site-A-only) and save them,
so the triptych can show FL vs centralized vs single-operator on one section."""
import numpy as np, torch, os
from fedseismic.config import RunConfig
from fedseismic.experiment import build_federated_run
from fedseismic.models import UNet
from fedseismic.privacy.log import load_meta
from fedseismic.federated.client import ClientTrainer

OUT = "results/runs/f2_compare"
os.makedirs(OUT, exist_ok=True)

def train_one(loaders, cfg, tag, epochs):
    m = UNet(1, cfg.num_classes, bilinear=False, norm=cfg.norm, groups=cfg.norm_groups)
    tr = ClientTrainer(m, loaders, torch.nn.CrossEntropyLoss(), device="cuda",
                       local_epochs=epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
                       optimizer=cfg.optimizer, momentum=cfg.momentum)
    tr.train()
    torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, f"{OUT}/{tag}.pt")
    print(f"  saved {tag}.pt", flush=True)

def main():
    run = "results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0"
    cfg = RunConfig.from_mapping(load_meta(run)["config"])
    data = build_federated_run(cfg, 0)
    # whole-F3: pool all sites' train loaders into one, train a single model
    from torch.utils.data import ConcatDataset, DataLoader
    pool = DataLoader(ConcatDataset([l.dataset for l in data.loaders]), batch_size=cfg.batch_size, shuffle=True)
    train_one([pool], cfg, "centralized_wholeF3", epochs=60)
    # site-A-only: train only on site 0's train loader
    train_one([data.loaders[0]], cfg, "siteA_only", epochs=60)
    print("F2 compare models done", flush=True)

if __name__ == "__main__":
    main()
