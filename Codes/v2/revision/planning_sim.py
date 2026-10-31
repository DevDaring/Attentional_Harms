"""
revision/planning_sim.py - prospective sample-size / precision simulation for the follow-up
human annotation (Abhinaba review s8-s9, D1, D2, retention non-inferiority).

This is a PLANNING TOOL. Nothing here is an empirical finding.

Scenarios
  S1  Matched absent/incidental replication (D1). For each sampled (family, model, draw) both the
      absent and the incidental B0 outputs are labelled. Families carry a random logit effect;
      2 matched pairs per family. Analysis: mean paired difference with a family-cluster
      bootstrap (percentile, 1,000 reps) and exact McNemar as a supplement.
  S2  Neutral-detail control (D1). Incidental identity vs incidental non-identity detail,
      same design as S1.
  S3  Retention: loss probability among required tasks, family-clustered; reports the
      probability that the one-sided 95% upper bound on the loss rate falls below a margin.
  S4  Diagnostic challenge set (D2): conditional repair rate among B0-positive outputs.

    python revision/planning_sim.py        -> revision/results/planning_sim.json, .csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from scipy.stats import beta, binomtest

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261024)


def cluster_boot_ci(d: np.ndarray, fam: np.ndarray, reps: int = 1000) -> tuple[float, float]:
    uniq, inv = np.unique(fam, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    cnts = np.bincount(inv)
    k = len(uniq)
    idx = RNG.integers(0, k, size=(reps, k))
    est = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def sim_paired(p0: float, p1: float, n_pairs: int, pairs_per_family: int = 2, sigma: float = 0.8,
               rho_within: float = 0.5, sims: int = 400) -> dict:
    """Paired binary outcomes with a shared family effect; returns power (cluster CI excludes 0)
    and mean CI half-width."""
    nf = int(np.ceil(n_pairs / pairs_per_family))
    hits, widths, mcn = 0, [], 0
    for _ in range(sims):
        u = RNG.normal(0, sigma, nf)
        fam = np.repeat(np.arange(nf), pairs_per_family)[:n_pairs]
        # within-pair shared component induces correlation between conditions
        e = RNG.normal(0, sigma * rho_within, n_pairs)
        y0 = RNG.random(n_pairs) < expit(logit(p0) + u[fam] + e)
        y1 = RNG.random(n_pairs) < expit(logit(p1) + u[fam] + e)
        d = y1.astype(float) - y0.astype(float)
        lo, hi = cluster_boot_ci(d, fam, 600)
        hits += lo > 0
        widths.append((hi - lo) / 2)
        b, c = int((y1 & ~y0).sum()), int((~y1 & y0).sum())
        mcn += (binomtest(min(b, c), b + c, 0.5).pvalue < 0.05) if b + c else 0
    return {"p0": p0, "p1": p1, "n_pairs": n_pairs, "families": nf, "sigma_logit": sigma,
            "power_cluster_ci_excludes_0": hits / sims, "power_mcnemar": mcn / sims,
            "mean_ci_halfwidth_pp": 100 * float(np.mean(widths))}


def sim_retention(loss: float, n_families: int, per_family: int = 1, margin: float = 0.10,
                  sigma: float = 0.8, sims: int = 2000) -> dict:
    ok = 0
    n = n_families * per_family
    for _ in range(sims):
        u = RNG.normal(0, sigma, n_families)
        p = expit(logit(loss) + np.repeat(u, per_family))
        k = int((RNG.random(n) < p).sum())
        # conservative: use family count as effective n when per_family > 1
        neff = n_families if per_family > 1 else n
        keff = k * neff / n
        ub = 1.0 if keff >= neff else beta.ppf(0.95, keff + 1, neff - keff)
        ok += ub < margin
    return {"true_loss": loss, "families": n_families, "per_family": per_family, "margin": margin,
            "prob_upper95_below_margin": ok / sims}


def sim_repair(repair: float, n_pos: int, sims: int = 4000) -> dict:
    w = []
    for _ in range(sims):
        k = RNG.binomial(n_pos, repair)
        lo = beta.ppf(0.025, k, n_pos - k + 1) if k else 0.0
        hi = beta.ppf(0.975, k + 1, n_pos - k) if k < n_pos else 1.0
        w.append((hi - lo) / 2)
    return {"true_repair_rate": repair, "b0_positive_units": n_pos, "mean_cp_halfwidth_pp": 100 * float(np.mean(w))}


def main() -> None:
    res = {"S1_incidental_vs_absent": [], "S2_identity_vs_neutral_detail": [], "S3_retention": [],
           "S4_repair": []}
    for p1 in (0.08, 0.13):
        for n in (80, 120, 160, 240, 320):
            for sigma in (0.5, 1.0):
                res["S1_incidental_vs_absent"].append(sim_paired(0.01, p1, n, 2, sigma))
    for p0, p1 in ((0.06, 0.13), (0.09, 0.13)):
        for n in (160, 240, 320, 480):
            res["S2_identity_vs_neutral_detail"].append(sim_paired(p0, p1, n, 2, 0.8))
    for loss in (0.0, 0.02, 0.05):
        for nf in (30, 45, 60, 90, 120):
            for margin in (0.10, 0.15):
                res["S3_retention"].append(sim_retention(loss, nf, 1, margin))
    for r in (0.3, 0.5, 0.7):
        for n in (20, 40, 60, 100):
            res["S4_repair"].append(sim_repair(r, n))
    res["zero_event_reference"] = {"one_sided_95_upper_zero_losses": {n: 1 - 0.05 ** (1 / n)
                                                                    for n in (9, 12, 30, 45, 59, 60, 90)}}
    with open(OUT / "planning_sim.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    for k, v in res.items():
        if isinstance(v, list):
            print("==", k)
            for r in v:
                print(r)


if __name__ == "__main__":
    main()
