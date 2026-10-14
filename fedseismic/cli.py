"""Command line interface for fedseismic experiments."""

import argparse
import json
from pathlib import Path

from .config import RunConfig
from .experiment import run
from .privacy.attacks import evaluate_run, flatten_attack_row
from .privacy.plot import plot_boundary
from .privacy.sweep import load_sweep_spec, run_sweep


def _print_run(cfg, results):
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
    if cfg.task != "classification":
        print(f"recovery rate: {results.recovery_rate:.6f}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m fedseismic.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--seed", type=int, action="append", dest="seeds")
    run_parser.add_argument("--output-dir")
    run_parser.add_argument("--device")

    attack_parser = subparsers.add_parser("attack")
    attack_parser.add_argument("--run-dir", required=True)
    attack_parser.add_argument("--device")
    attack_parser.add_argument("--json-out")

    sweep_parser = subparsers.add_parser("sweep")
    sweep_parser.add_argument("--config", required=True)
    sweep_parser.add_argument("--device")
    sweep_parser.add_argument("--skip-attack", action="store_true")

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--csv", required=True)
    plot_parser.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.command == "run":
        cfg = RunConfig.from_json(args.config)
        if args.output_dir:
            cfg.output_dir = args.output_dir
        if args.device:
            cfg.device = args.device
        results = run(cfg, args.seeds)
        return _print_run(cfg, results)
    if args.command == "attack":
        result = evaluate_run(args.run_dir, device=args.device)
        print(json.dumps(flatten_attack_row(result), indent=2))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    if args.command == "sweep":
        spec = load_sweep_spec(args.config)
        frame, jobs = run_sweep(spec, device=args.device, skip_attack=args.skip_attack)
        print(f"jobs: {len(jobs)}")
        if not frame.empty:
            print(frame.to_string(index=False))
        return frame
    if args.command == "plot":
        plot_boundary(args.csv, args.out)
        print(args.out)
        return args.out
    raise ValueError(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
