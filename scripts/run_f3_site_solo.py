"""Per-site solo baseline for the single-survey F3 grid.

Each F3 site trains alone on exactly the training sections it holds in the
federated run, with the federated run's configuration, seed, rounds, and local
epochs. The sites are built by the same code with the federated N, and then
only one site's loaders are kept, so the partition and holdout are identical.
Usage: run_f3_site_solo.py <N> <site> <seed> [--norm batch]
Output: results/runs/seismic_f3_site_solo/fedavg_n{N}_site{site}_{norm}/seed_{seed}
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fedseismic import experiment
from fedseismic.config import RunConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("site", type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("--norm", default="batch")
    args = parser.parse_args()

    source = ROOT / f"results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n{args.n}_{args.norm}norm/seed_{args.seed}/run_meta.json"
    config = json.loads(source.read_text())["config"]
    out = ROOT / f"results/runs/seismic_f3_site_solo/fedavg_n{args.n}_site{args.site}_{args.norm}/seed_{args.seed}"
    if (out / "results.json").is_file():
        print("done already", out)
        return
    config.update(
        num_clients=1, output_dir=str(out), device="cuda",
        privacy_log_last_only=True, privacy_save_local=False, privacy_save_received=False,
    )
    cfg = RunConfig.from_mapping(config)
    federated = dataclasses.replace(cfg, num_clients=args.n)
    original = experiment.build_federated_run

    def one_site(_cfg, seed):
        data = original(federated, seed)
        k = args.site
        return dataclasses.replace(
            data, train_parts=[data.train_parts[k]], loaders=[data.loaders[k]],
            client_tests=[data.client_tests[k]], client_info=[data.client_info[k]],
        )

    experiment.build_federated_run = one_site
    final, *_ = experiment._run_seed(cfg, args.seed)
    (out / "solo_meta.json").write_text(json.dumps({
        "federated_source": str(source.relative_to(ROOT)), "n": args.n, "site": args.site,
        "seed": args.seed, "norm": args.norm,
    }, indent=2))
    print("finished", out, "test1", final)


if __name__ == "__main__":
    main()
