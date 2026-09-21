"""Cartesian expansion and sequential runner for the privacy–personalization grid."""

from dataclasses import replace
from pathlib import Path
import json

import pandas as pd

from fedseismic.config import RunConfig
from fedseismic.experiment import run

from .attacks import evaluate_run, flatten_attack_row


def load_sweep_spec(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _job_name(job):
    alpha = job.get("partition_alpha")
    alpha_tag = f"a{alpha}" if alpha is not None else "aNA"
    lam = job.get("lambda_cap")
    lam_tag = f"l{lam}" if lam is not None else "lNA"
    diversity = job.get("fedkper_diversity") or "na"
    return (
        f"{job['algorithm']}_{job['split']}_{alpha_tag}_{lam_tag}_d{diversity}"
    )


def expand_jobs(spec):
    """Expand the paper grid without a full Cartesian of unused FedKPer knobs."""
    jobs = []
    heterogeneity = spec.get("heterogeneity") or [
        {"split": spec.get("split", "noniid"), "partition_alpha": spec.get("partition_alpha", 0.1)},
    ]
    algorithms = spec.get("algorithms") or ["fedavg"]
    lambda_caps = spec.get("lambda_cap") or [10.0]
    diversities = spec.get("fedkper_diversity") or ["infer"]
    oracle_reference = bool(spec.get("oracle_reference", False))
    for algorithm in algorithms:
        for het in heterogeneity:
            base = {
                "algorithm": algorithm,
                "split": het.get("split", "noniid"),
                "partition_alpha": het.get("partition_alpha", 0.1),
            }
            if algorithm == "fedkper":
                for lam in lambda_caps:
                    for diversity in diversities:
                        job = dict(base)
                        job["lambda_cap"] = lam
                        job["fedkper_diversity"] = diversity
                        job["agg_strategy"] = "fedkper"
                        jobs.append(job)
                if oracle_reference and "oracle" not in diversities:
                    job = dict(base)
                    job["lambda_cap"] = lambda_caps[-1]
                    job["fedkper_diversity"] = "oracle"
                    job["agg_strategy"] = "fedkper"
                    jobs.append(job)
            else:
                job = dict(base)
                job["lambda_cap"] = None
                job["fedkper_diversity"] = None
                if algorithm == "fedbn":
                    job["agg_strategy"] = "equal"
                elif algorithm == "fedprox":
                    job["agg_strategy"] = "equal"
                else:
                    job["agg_strategy"] = het.get("agg_strategy", "equal")
                jobs.append(job)
    for job in jobs:
        job["name"] = _job_name(job)
    return jobs


def _cell_config(base, spec, job, seed, output_dir, device=None):
    updates = {
        "algorithm": job["algorithm"],
        "split": job["split"],
        "partition_alpha": job["partition_alpha"],
        "agg_strategy": job["agg_strategy"],
        "seed": seed,
        "output_dir": str(output_dir),
    }
    if spec.get("num_rounds") is not None:
        updates["num_rounds"] = spec["num_rounds"]
    if spec.get("local_epochs") is not None:
        updates["local_epochs"] = spec["local_epochs"]
    if spec.get("num_clients") is not None:
        updates["num_clients"] = spec["num_clients"]
    if spec.get("sample_ratio") is not None:
        updates["sample_ratio"] = spec["sample_ratio"]
    if job.get("lambda_cap") is not None:
        updates["lambda_cap"] = job["lambda_cap"]
    if job.get("fedkper_diversity"):
        updates["fedkper_diversity"] = job["fedkper_diversity"]
    if device is not None:
        updates["device"] = device
    return replace(base, **updates)


def run_sweep(spec, device=None, skip_attack=False):
    base = RunConfig.from_json(spec["base"])
    output_root = Path(spec.get("output_root") or "results/privacy_boundary")
    seeds = list(spec.get("seeds") or [base.seed])
    jobs = expand_jobs(spec)
    rows = []
    for job in jobs:
        for seed in seeds:
            cell_dir = output_root / job["name"] / f"seed_{seed}"
            cfg = _cell_config(base, spec, job, seed, cell_dir, device=device)
            run(cfg, seeds=[seed])
            if skip_attack:
                continue
            result = evaluate_run(cell_dir)
            row = flatten_attack_row(result)
            row["job"] = job["name"]
            rows.append(row)
    frame = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "summary.csv"
    if not frame.empty:
        frame.to_csv(csv_path, index=False)
    return frame, jobs
