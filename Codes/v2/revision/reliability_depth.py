"""
revision/reliability_depth.py - reliability analysis for the TCSS manuscript.

For each annotation batch group: units, share of ratings marking an unnecessary mention, raw
pairwise agreement, Krippendorff's alpha with a unit bootstrap interval, Gwet's AC1, and
positive- and negative-specific agreement for the primary label. Alpha for a group is the
unit-weighted mean of batch alphas, as in the manuscript; the bootstrap resamples units
within each batch and recomputes that statistic.

Implements:
  Krippendorff, K. (2011). Computing Krippendorff's Alpha-Reliability. Annenberg School, UPenn.
  Gwet, K. L. (2008). Br. J. Math. Stat. Psychol., 61(1), 29-48.
  Cicchetti, D. V., & Feinstein, A. R. (1990). High agreement but low kappa: II. Resolving the
    paradoxes. J. Clin. Epidemiol., 43(6), 551-558 (positive/negative specific agreement).

    python revision/reliability_depth.py      -> revision/results/reliability_depth.json
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from analysis.agreement import gwet_ac1, krippendorff_alpha, raw_pairwise_agreement  # noqa: E402

SEED, B = 20261008, 2000
REP = V2 / "annotation" / "reports"
FUP = V2 / "revision" / "annotation_followup" / "reports"
GROUPS = {
    "pilot": [REP / "pilot"],
    "main_dev": [REP / b for b in ("main_1", "main_2", "main_3", "rgr2")],
    "comparators": [REP / "comparators"],
    "followup": [FUP / "FU1"],
}
FIELDS = {"unnecessary": ["yes", "no", "unclear"],
          "least_relevant": ["required", "useful_optional", "unnecessary", "unclear", "na"],
          "least_supported": ["explicit", "entailed", "unsupported", "unclear", "na"],
          "required_identity_retained": ["yes", "no", "unclear", "na"]}


def units_of(path: Path, field: str) -> list[list]:
    long = pd.read_csv(path / "raw_labels_long.csv", dtype=str, encoding="utf-8-sig")
    wide = long.pivot_table(index="unit_id", columns="rater", values=field, aggfunc="first")
    return [[v for v in row if isinstance(v, str) and v not in ("", "nan")] for row in wide.values.tolist()]


def specific_agreement(units: list[list], pos: str = "yes") -> tuple[float, float]:
    """Proportions of specific agreement over all rater pairs (Cicchetti and Feinstein, 1990)."""
    a = b = d = 0  # a: both pos, b: discordant, d: both neg
    for u in units:
        for x, y in combinations(u, 2):
            if x == pos and y == pos:
                a += 1
            elif x == pos or y == pos:
                b += 1
            else:
                d += 1
    ppos = 2 * a / (2 * a + b) if (2 * a + b) else float("nan")
    pneg = 2 * d / (2 * d + b) if (2 * d + b) else float("nan")
    return ppos, pneg


def weighted_alpha(batches: list[list[list]]) -> float:
    n = [len(u) for u in batches]
    return float(sum(krippendorff_alpha(u) * k for u, k in zip(batches, n)) / sum(n))


rng = np.random.default_rng(SEED)
out = {"seed": SEED, "replicates": B, "groups": {}}
for g, paths in GROUPS.items():
    res = {}
    for f, cats in FIELDS.items():
        batches = [units_of(p, f) for p in paths]
        pooled = [u for bt in batches for u in bt]
        r = {"units": len(pooled), "alpha": weighted_alpha(batches),
             "raw_agreement": raw_pairwise_agreement(pooled), "ac1": gwet_ac1(pooled, cats)}
        if f == "unnecessary":
            ratings = [v for u in pooled for v in u]
            r["share_yes"] = ratings.count("yes") / len(ratings)
            r["ppos"], r["pneg"] = specific_agreement(pooled)
            boots = []
            for _ in range(B):
                rs = [[bt[i] for i in rng.integers(0, len(bt), len(bt))] for bt in batches]
                boots.append(weighted_alpha(rs))
            r["alpha_ci"] = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
        res[f] = r
    out["groups"][g] = res
    u = res["unnecessary"]
    print(f"{g:12s} n={u['units']:4d} yes={100*u['share_yes']:.1f}% agr={u['raw_agreement']:.3f} "
          f"alpha={u['alpha']:.2f} [{u['alpha_ci'][0]:.2f},{u['alpha_ci'][1]:.2f}] ac1={u['ac1']:.2f} "
          f"ppos={u['ppos']:.2f} pneg={u['pneg']:.3f} | rel={res['least_relevant']['alpha']:.2f} "
          f"sup={res['least_supported']['alpha']:.2f} ret={res['required_identity_retained']['alpha']:.2f}")

(V2 / "revision" / "results" / "reliability_depth.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
