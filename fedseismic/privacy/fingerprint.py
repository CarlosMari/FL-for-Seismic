"""Peer cohort fingerprint from zero patterns.

Sub-task of workstream C (handoff section 6): 10 clients, 3 per round, the
schedule hidden from peers. Can a peer recover its cohort — which other
clients shared its rounds — from the updates it sees?

Observer is peer-sum: in each round the peer participates in, it sees the
round sum and subtracts its own update, leaving the residual (sum of the
other members' updates, plaintext simulation, no crypto). The schedule is
hidden: true cohorts are used only for scoring, never for the attack.

Attack: the residual's support pattern fingerprints the cohort. For every
pair of the peer's round-observations, predict same-cohort from the distance
between residual channel vectors; the metric is the pairwise cohort-membership
AUC (0.5 = chance). No clustering-k to tune, no labels used to fit anything:
the channels are hand-built (bias delta, row sizes, Decaf positive sum) on
rounds 1..10, the regime where the count slope lives (see sum_channel).
Method comparison (FedAvg vs FedLC) uses the same channels both sides.
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from fedseismic.privacy.decaf import positive_sum
from fedseismic.privacy.head_channels import _client_id
from fedseismic.privacy.log import load_meta

OBSERVER = "peer-sum"
ROUND_LO = 1
ROUND_HI = 10


def round_residuals(run_dir, round_lo=ROUND_LO, round_hi=ROUND_HI):
    """Per (round, peer): residual channel vectors = mean of other members' deltas.

    Each head file is upload-minus-received for one participation, so the
    residual over the other members is the plaintext stand-in for what a peer
    computes from the encrypted sum minus its own update.
    """
    by_round = {}
    for path in sorted((Path(run_dir) / "head").glob("round_*_client_*.npz")):
        round_index = int(path.stem.split("_")[1])
        if not (round_lo <= round_index <= round_hi):
            continue
        with np.load(path) as handle:
            weight = np.asarray(handle["weight_delta"], dtype=np.float64)
            bias_delta = np.asarray(handle["bias_delta"], dtype=np.float64).reshape(-1)
        by_round.setdefault(round_index, {})[_client_id(path)] = (
            np.linalg.norm(weight.reshape(weight.shape[0], -1), axis=1),
            bias_delta,
            positive_sum(weight),
        )
    observations = []
    for round_index, members in by_round.items():
        clients = sorted(members)
        for peer in clients:
            others = [client for client in clients if client != peer]
            if not others:
                continue
            rows = np.stack([members[client][0] for client in others])
            biases = np.stack([members[client][1] for client in others])
            decafs = np.stack([members[client][2] for client in others])
            observations.append({
                "round": round_index,
                "peer": peer,
                "cohort": tuple(clients),
                "others": tuple(others),
                "row": rows.mean(axis=0),
                "bias": biases.mean(axis=0),
                "decaf": decafs.mean(axis=0),
            })
    return observations


def pairwise_cohort_auc(observations, channel):
    """AUC of same-cohort prediction from residual distances, per peer pooled.

    Pairs are formed within each peer's own observations (a peer never sees
    another peer's residuals). Distance is cosine distance on the channel
    vector. Same-cohort pairs are positives.
    """
    from collections import defaultdict

    by_peer = defaultdict(list)
    for obs in observations:
        by_peer[obs["peer"]].append(obs)
    labels = []
    scores = []
    for peer, obs in by_peer.items():
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                same = 1 if obs[i]["cohort"] == obs[j]["cohort"] else 0
                left = np.asarray(obs[i][channel], dtype=np.float64)
                right = np.asarray(obs[j][channel], dtype=np.float64)
                denom = float(np.linalg.norm(left) * np.linalg.norm(right))
                cosine = float(np.dot(left, right) / denom) if denom > 1e-18 else 0.0
                labels.append(same)
                scores.append(cosine)
    labels = np.asarray(labels)
    if len(labels) < 4 or len(np.unique(labels)) < 2:
        return float("nan"), 0
    return float(roc_auc_score(labels, np.asarray(scores))), int(len(labels))


def score_run(run_dir, round_lo=ROUND_LO, round_hi=ROUND_HI):
    """Pairwise cohort AUC per channel for one seed. No fitting, no labels."""
    observations = round_residuals(run_dir, round_lo, round_hi)
    out = {"n_observations": len(observations)}
    for channel in ("bias", "row", "decaf"):
        auc, pairs = pairwise_cohort_auc(observations, channel)
        out[f"{channel}_cohort_auc"] = auc
        out[f"{channel}_pairs"] = pairs
    peers = sorted({obs["peer"] for obs in observations})
    out["n_peers"] = len(peers)
    return out


def score_grid(roots, n_draws=1000, output="results/tables/fingerprint_blood.csv",
               round_lo=ROUND_LO, round_hi=ROUND_HI):
    """Score finished fingerprint runs. Bootstrap resamples seeds."""
    import pandas as pd

    from fedseismic.config import RunConfig
    from fedseismic.privacy.log import load_meta
    from fedseismic.privacy.mechanism import DISPLAY, _mean_finite, _percentile

    found = []
    for root in roots:
        for job in sorted(Path(root).iterdir()):
            if not job.is_dir():
                continue
            for run in sorted(job.glob("seed_*")):
                if (run / "results.json").is_file() and (run / "head").is_dir():
                    found.append(run)
    if not found:
        raise FileNotFoundError(f"no finished runs under {list(roots)}")
    per_seed = {}
    for run in found:
        meta = load_meta(run)
        cfg = RunConfig.from_mapping(meta["config"])
        print(f"score {run}", flush=True)
        scored = score_run(run, round_lo, round_hi)
        key = (str(cfg.dataset), run.parent.name, int(cfg.num_clients),
               float(cfg.partition_alpha), str(cfg.algorithm), bool(cfg.logit_adjust))
        per_seed.setdefault(key, []).append(scored)
    table = []
    rng = np.random.RandomState(0)
    for key in sorted(per_seed):
        dataset, job, n_members, alpha, algorithm, logit = key
        seeds = per_seed[key]
        method = "FedLC" if logit else "FedAvg"
        for channel in ("bias", "row", "decaf"):
            values = [row[f"{channel}_cohort_auc"] for row in seeds]
            values = [value for value in values if np.isfinite(value)]
            if not values:
                continue
            point = _mean_finite(values)
            draws = [_mean_finite([values[item] for item in rng.randint(0, len(values), size=len(values))])
                     for _ in range(n_draws)]
            low, high = _percentile(draws)
            pairs = int(np.mean([row[f"{channel}_pairs"] for row in seeds]))
            table.append({
                "dataset": DISPLAY.get(dataset, dataset),
                "job": job,
                "n_members": n_members,
                "alpha": alpha,
                "method": method,
                "observer": OBSERVER,
                "channel": channel,
                "metric": "cohort_auc",
                "estimate": point,
                "low": low,
                "high": high,
                "floor": 0.5,
                "n_seeds": len(values),
                "pairs_per_seed": pairs,
            })
            print(f"  {method} {channel} cohort_auc {point:.3f} [{low:.3f}, {high:.3f}]", flush=True)
    frame = pd.DataFrame(table)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {path.resolve()}")
    return table


def main_cli(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Peer cohort fingerprint scoring")
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", default="results/tables/fingerprint_blood.csv")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--round-lo", type=int, default=ROUND_LO)
    parser.add_argument("--round-hi", type=int, default=ROUND_HI)
    args = parser.parse_args(argv)
    return score_grid(args.roots, n_draws=args.draws, output=args.output,
                      round_lo=args.round_lo, round_hi=args.round_hi)


if __name__ == "__main__":
    main_cli()
