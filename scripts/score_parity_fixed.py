"""Score the corrected two-survey runs without overwriting legacy tables."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("parity_fl_fixed", "parity_f3_solo_fixed", "parity_pari_solo_fixed")
BLOCKS = {"A": ("F3", 0, 6), "B": ("Parihaka", 6, 12)}


def main():
    from fedseismic.privacy.multisurvey_utility import rescore_run

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="results/tables")
    args = parser.parse_args()
    summaries, details = [], []
    for arm in ARMS:
        for seed in (0, 1):
            matches = list((Path("results/runs") / arm).glob(f"*/seed_{seed}"))
            if len(matches) != 1 or not (matches[0] / "results.json").is_file():
                raise RuntimeError(f"expected one completed run for {arm}, seed {seed}")
            run = matches[0]
            print(f"Rescoring {run}", flush=True)
            score = rescore_run(run, device=args.device)
            priors = {row["client"]: row["pi"] for row in json.loads(
                (run / "labels/pi.json").read_text())["clients"]}
            for code, (survey, lo, hi) in BLOCKS.items():
                if not np.isfinite(score[f"{code}_local_miou"]):
                    continue
                classes = [r for r in score["_per_site_class_rows"]
                           if r["survey"] == code and lo <= r["class"] < hi]
                if len(classes) != hi - lo:
                    raise RuntimeError(f"unexpected class count for {run}, {survey}")
                seen_values = []
                for r in classes:
                    seen = priors[r["client"]][r["class"]] > 0
                    details.append({"arm": arm, "seed": seed, "survey": survey,
                                    "union_class": r["class"], "iou": r["iou"],
                                    "test_present": r["present"], "train_present": seen})
                    if seen and r["present"]:
                        seen_values.append(r["iou"])
                summaries.append({"arm": arm, "seed": seed, "survey": survey,
                                  "present_class_miou": score[f"{code}_local_miou"],
                                  "survey_all_class_miou": np.mean([r["iou"] for r in classes]),
                                  "train_seen_test_present_miou": np.mean(seen_values) if seen_values else np.nan})
    frame = pd.DataFrame(summaries)
    for survey, solo in (("F3", "parity_f3_solo_fixed"),
                         ("Parihaka", "parity_pari_solo_fixed")):
        for seed in (0, 1):
            fl = frame[(frame.arm == "parity_fl_fixed") & (frame.seed == seed) & (frame.survey == survey)]
            baseline = frame[(frame.arm == solo) & (frame.seed == seed) & (frame.survey == survey)]
            if len(fl) != 1 or len(baseline) != 1:
                raise RuntimeError(f"missing comparison for {survey}, seed {seed}")
            delta = fl.iloc[0].present_class_miou - baseline.iloc[0].present_class_miou
            print(f"{survey} seed {seed}: FL - solo = {delta:+.6f}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "parity_fixed_runs.csv", index=False)
    pd.DataFrame(details).to_csv(out / "parity_fixed_per_class.csv", index=False)
    print(f"Wrote corrected scores and class support to {out}")


if __name__ == "__main__":
    main()
