"""Command line interface for fedseismic experiments."""

import argparse

from .config import RunConfig
from .experiment import run


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m fedseismic.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--seed", type=int, action="append", dest="seeds")
    args = parser.parse_args(argv)
    if args.command == "run":
        cfg = RunConfig.from_json(args.config)
        results = run(cfg, args.seeds)
        mean, std = results.mean_std
        local_mean, local_std = results.local_mean_std
        metric = "acc" if cfg.task == "classification" else "mIoU"
        print(f"device: {cfg.device}")
        print(f"final {metric}: {results.miou_final}")
        print(f"mean +/- std: {mean:.6f} +/- {std:.6f}")
        print(f"local {metric}: {results.local_mean} (worst {results.local_worst})")
        print(f"local mean +/- std: {local_mean:.6f} +/- {local_std:.6f}")
        print(f"global-on-local: {results.global_on_local}")
        print(f"balance: {results.balance:.6f}")
        print(f"recovery rate: {results.recovery_rate:.6f}")
        return results


if __name__ == "__main__":
    main()
