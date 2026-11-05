"""Ch6 H10 by round: the segmentation-head bias channel in rounds 1-5 vs round 20.

Early: results/runs/seismic_f3_early_heads (5-round reruns of the F3 grid with
outc deltas logged every round, head/). Round 20: results/runs/seismic_f3
head_outc/. Same 8 jobs, same seeds. Per N and round: Zechstein sign rule,
balanced accuracy of "present iff bias change > 0" over all site-facies pairs,
and the mean within-facies Pearson (log count vs bias), averaged over jobs.
CPU only; writes figures/ch6/data/early_heads.csv.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from fedseismic.privacy.seismic_channel import ABSENT, site_counts, _within_pearson
from fedseismic.privacy.log import load_labels

EARLY = ROOT / "results/runs/seismic_f3_early_heads"
LATE = ROOT / "results/runs/seismic_f3"
OUT = ROOT / "figures/ch6/data/early_heads.csv"


def biases(seed_dir, dirname, rnd):
    out = {}
    for p in sorted((seed_dir / dirname).glob(f"round_{rnd:03d}_client_*.npz")):
        with np.load(p) as h:
            out[int(p.stem.split("_")[-1])] = np.asarray(h["bias_delta"], float).reshape(-1)
    return out


def main():
    rows = []
    for job in sorted(EARLY.glob("*")):
        n = int(job.name.split("_dna_n")[1].split("_")[0])
        for seed_dir in sorted(job.glob("seed_*")):
            if not (seed_dir / "results.json").is_file():
                continue
            late_dir = LATE / job.name / seed_dir.name
            hist, _ = load_labels(seed_dir)
            counts = site_counts(seed_dir)
            for rnd, src, dirname in [(r, seed_dir, "head") for r in range(1, 6)] + [(20, late_dir, "head_outc")]:
                b = biases(src, dirname, rnd)
                if not b:
                    continue
                cl = sorted(b)
                B = np.stack([b[c] for c in cl])
                P = np.stack([np.asarray(hist[c], float) / np.sum(hist[c]) for c in cl]) > ABSENT
                L = np.log10(np.maximum(counts[cl], 1.0))
                rows.append({"job": job.name, "n": n, "seed": seed_dir.name, "round": rnd,
                             "zech_absent": int((~P[:, 5]).sum()), "zech_absent_neg": int(((~P[:, 5]) & (B[:, 5] < 0)).sum()),
                             "zech_present": int(P[:, 5].sum()), "zech_present_pos": int((P[:, 5] & (B[:, 5] > 0)).sum()),
                             "tp": int((P & (B > 0)).sum()), "pos": int(P.sum()),
                             "tn": int((~P & (B <= 0)).sum()), "neg": int((~P).sum()),
                             "L": L, "B": B})
    f = pd.DataFrame(rows)
    summary = []
    for (n, rnd), g in f.groupby(["n", "round"]):
        tpr, tnr = g.tp.sum() / g.pos.sum(), g.tn.sum() / max(g.neg.sum(), 1)
        rs = []
        for job, gj in g.groupby("job"):  # pool seeds within a job, as the thesis does
            Bj = np.concatenate(list(gj.B)); Bj = Bj / Bj.std()
            rs.append(_within_pearson(np.concatenate(list(gj.L)), Bj))
        summary.append({"n": n, "round": rnd, "runs": len(g),
                        "zech_absent_neg": f"{g.zech_absent_neg.sum()}/{g.zech_absent.sum()}",
                        "zech_present_pos": f"{g.zech_present_pos.sum()}/{g.zech_present.sum()}",
                        "balanced_acc": round((tpr + tnr) / 2, 3), "tpr_present": round(tpr, 3),
                        "within_r_mean": round(float(np.nanmean(rs)), 3),
                        "within_r_min": round(float(np.nanmin(rs)), 3), "within_r_max": round(float(np.nanmax(rs)), 3)})
    s = pd.DataFrame(summary)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT, index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
