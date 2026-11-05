"""Support leakage from ONE upload versus the client's upload history (reviewer point 2.2).

Same runs, channels, AUC, floor, and bootstrap as Table 5.1 (prior_smooth_* grids,
p1 = FedAvg, p0 = LA; 5 seeds; clients resampled inside every seed). Only the
observation changes:
  history : channels averaged over all of a client's participations (as in Table 5.1)
  first   : the client's first participation only (one preselected upload per client)
  band    : the client's first participation inside a round band (one per client per band)
No shadows are needed: presence AUC scores the raw channel.
Writes figures/single_upload/data/support_single_vs_history.csv.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from fedseismic.privacy.decaf import positive_sum
from fedseismic.privacy.head_channels import _client_id
from fedseismic.privacy.log import load_labels
from fedseismic.privacy.mechanism import iter_runs, bootstrap_seed_mean, prior_presence_auc, DISPLAY
from fedseismic.privacy.metrics import to_simplex
from fedseismic.privacy.row_delta import _presence_auc

BANDS = {"1-5": (1, 5), "6-10": (6, 10), "11-20": (11, 20), "21-40": (21, 40)}
CHANNELS = ("bias", "row", "decaf")


def participations(run_dir):
    """client -> list of (round, channels) in round order."""
    out = {}
    for path in sorted((Path(run_dir) / "head").glob("round_*_client_*.npz")):
        rnd = int(path.stem.split("_")[1])
        with np.load(path) as h:
            w = np.asarray(h["weight_delta"], dtype=np.float64)
            b = np.asarray(h["bias_delta"], dtype=np.float64).reshape(-1)
        out.setdefault(_client_id(path), []).append(
            (rnd, {"bias": b, "row": np.linalg.norm(w.reshape(w.shape[0], -1), axis=1), "decaf": positive_sum(w)}))
    for c in out:
        out[c].sort(key=lambda t: t[0])
    return out


def rows_for(run_dir, view):
    hist, _ = load_labels(run_dir)
    parts = participations(run_dir)
    rows = []
    for c in sorted(parts):
        if c not in hist:
            continue
        p = parts[c]
        if view == "history":
            pick = p
        elif view == "first":
            pick = p[:1]
        else:
            lo, hi = BANDS[view]
            pick = [t for t in p if lo <= t[0] <= hi][:1]
        if not pick:
            continue
        row = {"client": c, "pi": to_simplex(hist[c]), "round": pick[0][0], "n_obs": len(pick)}
        for ch in CHANNELS:
            row[ch] = np.mean(np.stack([t[1][ch] for t in pick]), axis=0)
        rows.append(row)
    return rows


def auc_of(ch):
    return lambda rows: _presence_auc([r["pi"] for r in rows], [r[ch] for r in rows])


def main():
    runs = [r for r in iter_runs() if r["prior_smooth"] in (0.0, 1.0)]
    out = []
    for view in ("history", "first", *BANDS):
        grouped = {}
        for r in runs:
            grouped.setdefault((r["dataset"], r["prior_smooth"]), []).append(rows_for(r["path"], view))
        for (ds, ps), groups in sorted(grouped.items()):
            method = "FedAvg" if ps == 1.0 else "LA"
            floor = bootstrap_seed_mean(groups, prior_presence_auc, n_draws=1000)[0]
            n = int(np.mean([len(g) for g in groups]))
            rnd = float(np.mean([r["round"] for g in groups for r in g]))
            for ch in CHANNELS:
                est, lo, hi, _ = bootstrap_seed_mean(groups, auc_of(ch), n_draws=1000)
                out.append({"dataset": DISPLAY.get(ds, ds), "method": method, "view": view, "channel": ch,
                            "auc": round(est, 4), "low": round(lo, 4), "high": round(hi, 4),
                            "floor": round(floor, 4), "clients_per_seed": n, "mean_round": round(rnd, 1)})
                print(out[-1], flush=True)
    f = pd.DataFrame(out)
    dest = ROOT / "figures/single_upload/data"
    dest.mkdir(parents=True, exist_ok=True)
    f.to_csv(dest / "support_single_vs_history.csv", index=False)


if __name__ == "__main__":
    main()
