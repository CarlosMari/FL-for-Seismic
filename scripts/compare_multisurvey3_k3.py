"""Per-survey comparison of the one-client-per-survey runs.

Rescores saved checkpoints on each survey's buffered holdout (no training) and
writes one row per arm, seed and survey, with FL - solo paired by seed. Seeds
are reported individually downstream: with so few, the spread is a finding and
not an error bar.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = {
    "solo": ("multisurvey3_k3_solo_a", "multisurvey3_k3_solo_b", "multisurvey3_k3_solo_c"),
    "fedpersurvey": ("multisurvey3_k3_fedpersurvey",),
    "fedhca2": ("multisurvey3_k3_fedhca2",),
}
SURVEYS = {"A": "F3", "B": "Parihaka", "C": "Penobscot"}
CLASS_BLOCKS = {"A": (0, 6), "B": (6, 12), "C": (12, 19)}


def runs_for(arm, runs_root, fedhca2_root=None):
    roots = (fedhca2_root,) if arm == "fedhca2" and fedhca2_root else ARMS[arm]
    for root in roots:
        for run in sorted((Path(runs_root) / root).glob("*/seed_*")):
            if (run / "results.json").is_file():
                yield run


def rows_from_score(arm, seed, run, scored):
    """One row per survey that this run actually evaluated."""
    rows = []
    for code, survey in SURVEYS.items():
        if np.isnan(scored[f"{code}_local_miou"]):
            continue
        lo, hi = CLASS_BLOCKS[code]
        site_rows = [row for row in scored["_per_site_class_rows"]
                     if row["survey"] == code and lo <= row["class"] < hi]
        # The scorer's "all" value averages over all 19 union rows. Compute
        # the survey's own 6/6/7-class mean separately (including absent=0).
        if len(site_rows) != hi - lo:
            raise ValueError(f"expected {hi - lo} class rows for {survey}, got {len(site_rows)}")
        survey_all = float(np.mean([row["iou"] for row in site_rows]))
        rows.append({
            "arm": arm, "seed": seed, "survey": survey, "run": str(run),
            "miou_all_class": survey_all,
            "miou_union_all_class": scored[f"{code}_local_all_miou"],
            "miou_present_class": scored[f"{code}_local_miou"],
        })
    return rows


def add_solo_delta(frame):
    """Attach FL - solo (all-class), paired by survey and seed."""
    solo = frame[frame["arm"] == "solo"][["survey", "seed", "miou_all_class"]]
    solo = solo.rename(columns={"miou_all_class": "solo_all_class"})
    merged = frame.merge(solo, on=["survey", "seed"], how="left")
    merged["delta_vs_solo"] = merged["miou_all_class"] - merged["solo_all_class"]
    merged.loc[merged["arm"] == "solo", "delta_vs_solo"] = np.nan
    return merged.drop(columns="solo_all_class")


def summarize(frame):
    return (
        frame.groupby(["survey", "arm"], sort=True)
        .agg(n_seeds=("seed", "nunique"),
             miou_all_class=("miou_all_class", "mean"),
             seed_std=("miou_all_class", "std"),
             miou_present_class=("miou_present_class", "mean"),
             delta_vs_solo=("delta_vs_solo", "mean"))
        .reset_index()
    )


def main(argv=None):
    from fedseismic.privacy.multisurvey_utility import rescore_run

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--out-dir", default="results/tables")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fedhca2-root", default=None,
                        help="run-root name for a non-overwriting FedHCA2 restart")
    args = parser.parse_args(argv)

    rows, per_class = [], []
    for arm in ARMS:
        for run in runs_for(arm, args.runs_root, args.fedhca2_root):
            seed = json.loads((run / "run_meta.json").read_text())["seed"]
            print(f"rescore {arm} {run}", flush=True)
            scored = rescore_run(run, device=args.device)
            rows.extend(rows_from_score(arm, seed, run, scored))
            for row in scored["_per_site_class_rows"]:
                per_class.append({"arm": arm, "seed": seed, **row})
    if not rows:
        raise SystemExit(f"no finished runs under {args.runs_root}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = add_solo_delta(pd.DataFrame(rows))
    frame.to_csv(out / "multisurvey3_k3_runs.csv", index=False)
    pd.DataFrame(per_class).to_csv(out / "multisurvey3_k3_per_class.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(out / "multisurvey3_k3_summary.csv", index=False)
    print(frame.drop(columns="run").round(3).to_string(index=False))
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
