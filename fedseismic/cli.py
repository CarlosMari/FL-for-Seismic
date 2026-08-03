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
        results = run(RunConfig.from_json(args.config), args.seeds)
        mean, std = results.mean_std
        print(f"final mIoU: {results.miou_final}")
        print(f"mean +/- std: {mean:.6f} +/- {std:.6f}")
        print(f"recovery rate: {results.recovery_rate:.6f}")
        return results


if __name__ == "__main__":
    main()
