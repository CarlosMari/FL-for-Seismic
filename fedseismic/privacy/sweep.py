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
    name = (
        f"{job['algorithm']}_{job['split']}_{alpha_tag}_{lam_tag}_d{diversity}"
    )
    if job.get("dp_noise_multiplier") is not None:
        name = f"{name}_s{job['dp_noise_multiplier']}"
    balance = job.get("gradient_balance")
    if balance and (balance != "example" or job.get("local_steps")):
        step_tag = f"_s{job['local_steps']}" if job.get("local_steps") else ""
        name = f"{name}_b{balance}{step_tag}"
    if job.get("num_clients") is not None:
        name = f"{name}_n{int(job['num_clients'])}"
    if job.get("sample_ratio") is not None and abs(float(job["sample_ratio"]) - 1.0) > 1e-9:
        name = f"{name}_r{float(job['sample_ratio']):g}"
    return name


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
    silo_counts = spec.get("num_clients_list") or (
        [spec["num_clients"]] if spec.get("num_clients") is not None else [None]
    )
    sample_ratios = spec.get("sample_ratio_list") or (
        [spec["sample_ratio"]] if spec.get("sample_ratio") is not None else [None]
    )
    for algorithm in algorithms:
        for het in heterogeneity:
            for silos in silo_counts:
                for ratio in sample_ratios:
                    base = {
                        "algorithm": algorithm,
                        "split": het.get("split", "noniid"),
                        "partition_alpha": het.get("partition_alpha", 0.1),
                    }
                    if silos is not None:
                        base["num_clients"] = int(silos)
                        base["sample_ratio"] = 1.0
                    if ratio is not None:
                        base["sample_ratio"] = float(ratio)
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
                        noises = spec.get("dp_noise_multiplier") if algorithm == "dpfedavg" else [None]
                        if not isinstance(noises, list):
                            noises = [noises]
                        for noise in noises:
                            job = dict(base)
                            job["lambda_cap"] = None
                            job["fedkper_diversity"] = None
                            if algorithm == "fedbn":
                                job["agg_strategy"] = "equal"
                            elif algorithm == "fedprox":
                                job["agg_strategy"] = "equal"
                            else:
                                job["agg_strategy"] = het.get("agg_strategy", "equal")
                            if noise is not None:
                                job["dp_noise_multiplier"] = noise
                                job["dp_clip_norm"] = spec.get("dp_clip_norm", 5.0)
                            jobs.append(job)
    for job in jobs:
        job["name"] = _job_name(job)
    jobs = _expand_steps(jobs, spec)
    norms = spec.get("norm_list")
    if isinstance(norms, list):
        multiplied = []
        for job in jobs:
            for norm in norms:
                copy = dict(job)
                copy["norm"] = str(norm)
                copy["name"] = f"{job['name']}_{str(norm)}norm"
                multiplied.append(copy)
        jobs = multiplied
    smooth = spec.get("prior_smooth")
    smooth = spec.get("prior_smooth")
    if isinstance(smooth, list):
        multiplied = []
        for job in jobs:
            for value in smooth:
                copy = dict(job)
                copy["prior_smooth"] = float(value)
                copy["name"] = f"{job['name']}_p{float(value):g}"
                multiplied.append(copy)
        return multiplied
    return jobs


def _expand_steps(jobs, spec):
    """One job per full-dataset step count and per gradient weighting."""
    steps = spec.get("local_steps")
    balances = spec.get("gradient_balance")
    if not steps and not balances:
        return jobs
    if not isinstance(steps, list):
        steps = [steps]
    if not isinstance(balances, list):
        balances = [balances] if balances else ["example"]
    multiplied = []
    for job in jobs:
        for balance in balances:
            for step in steps:
                copy = dict(job)
                if step is not None:
                    copy["local_steps"] = int(step)
                copy["gradient_balance"] = str(balance)
                copy["infer_log_prior"] = str(balance) in {"class", "class_minibatch"}
                copy["name"] = _job_name(copy)
                multiplied.append(copy)
    return multiplied


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
    # Job-level silo counts and participation override spec-level defaults.
    if spec.get("num_clients") is not None and job.get("num_clients") is None:
        updates["num_clients"] = spec["num_clients"]
    if spec.get("sample_ratio") is not None and job.get("sample_ratio") is None:
        updates["sample_ratio"] = spec["sample_ratio"]
    if job.get("num_clients") is not None:
        updates["num_clients"] = int(job["num_clients"])
    if job.get("sample_ratio") is not None:
        updates["sample_ratio"] = float(job["sample_ratio"])
    if spec.get("privacy_round_batches") is not None:
        updates["privacy_round_batches"] = spec["privacy_round_batches"]
    if spec.get("ditto_lambda") is not None:
        updates["ditto_lambda"] = spec["ditto_lambda"]
    if spec.get("privacy_log_last_only") is not None:
        updates["privacy_log_last_only"] = spec["privacy_log_last_only"]
    if spec.get("logit_adjust") is not None:
        updates["logit_adjust"] = spec["logit_adjust"]
    if job.get("prior_smooth") is not None:
        updates["prior_smooth"] = float(job["prior_smooth"])
    if spec.get("privacy_save_received") is not None:
        updates["privacy_save_received"] = spec["privacy_save_received"]
    if spec.get("class_reweight") is not None:
        updates["class_reweight"] = spec["class_reweight"]
    if spec.get("public_pad_per_class") is not None:
        updates["public_pad_per_class"] = spec["public_pad_per_class"]
    if spec.get("train_public_pad") is not None:
        updates["train_public_pad"] = spec["train_public_pad"]
    if spec.get("presence_rehearsal") is not None:
        updates["presence_rehearsal"] = spec["presence_rehearsal"]
    if spec.get("spatial_holdout_buffer") is not None:
        updates["spatial_holdout_buffer"] = int(spec["spatial_holdout_buffer"])
    if job.get("dp_noise_multiplier") is not None:
        updates["dp_noise_multiplier"] = job["dp_noise_multiplier"]
    if job.get("dp_clip_norm") is not None:
        updates["dp_clip_norm"] = job["dp_clip_norm"]
    if job.get("lambda_cap") is not None:
        updates["lambda_cap"] = job["lambda_cap"]
    if job.get("fedkper_diversity"):
        updates["fedkper_diversity"] = job["fedkper_diversity"]
    if job.get("local_steps") is not None:
        updates["local_steps"] = int(job["local_steps"])
    if job.get("gradient_balance"):
        updates["gradient_balance"] = job["gradient_balance"]
    if job.get("infer_log_prior") is not None:
        updates["infer_log_prior"] = bool(job["infer_log_prior"])
    if job.get("norm") is not None:
        updates["norm"] = str(job["norm"])
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
            if (cell_dir / "results.json").is_file():
                print(f"skip {job['name']} seed {seed}", flush=True)
                continue
            print(f"running {job['name']} seed {seed}", flush=True)
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
