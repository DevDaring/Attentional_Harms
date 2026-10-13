"""
analysis/power.py - post-pilot power / precision simulation (Future_Plan.md s6).

Simulates a paired family-level design: each family contributes m responses
per arm (models x draws) with a family random intercept on the logit scale.
The interval for the arm difference uses family-level paired differences, so
repeated draws never count as independent scenarios.

Inputs come from the pilot: baseline prevalence p0 and between-family SD
sigma (estimated here from pilot labels by the method of moments on family
means). Treat the output as planning guidance, not as a guarantee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit


def estimate_sigma(labels: pd.DataFrame, y: str = "y", family: str = "family_id") -> float:
    """Between-family SD on the logit scale from family means (moment method)."""
    g = labels.groupby(family)[y].agg(["mean", "count"])
    p = labels[y].mean()
    if p <= 0 or p >= 1 or len(g) < 3:
        return 0.5
    within = (p * (1 - p) / g["count"]).mean()
    between = max(g["mean"].var(ddof=1) - within, 1e-6)
    return float(np.sqrt(between) / (p * (1 - p)))       # delta-method to logit scale


def simulate(p0: float, delta: float, n_families: int, m_per_arm: int, sigma: float,
             reps: int = 2000, seed: int = 1) -> dict:
    """delta = p1 - p0 (e.g. -0.10 for a 10-point reduction)."""
    rng = np.random.default_rng(seed)
    p1 = min(max(p0 + delta, 1e-4), 1 - 1e-4)
    l0, l1 = logit(max(p0, 1e-4)), logit(p1)
    half, hits = [], 0
    for _ in range(reps):
        u = rng.normal(0, sigma, n_families)
        y0 = rng.binomial(m_per_arm, expit(l0 + u)) / m_per_arm
        y1 = rng.binomial(m_per_arm, expit(l1 + u)) / m_per_arm
        d = y1 - y0
        se = d.std(ddof=1) / np.sqrt(n_families)
        lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
        half.append(1.96 * se)
        hits += int(hi < 0) if delta < 0 else int(lo > 0)
    return {"p0": p0, "p1": p1, "delta": delta, "n_families": n_families, "m_per_arm": m_per_arm,
            "sigma_logit": sigma, "mean_ci_half_width": float(np.mean(half)),
            "power": hits / reps, "reps": reps}


def grid(p0: float, sigma: float, relative=(0.25, 0.50, 0.75), families=(40, 80, 120, 160),
         m_per_arm: int = 12) -> pd.DataFrame:
    """Relative reductions (delta = -r * p0). An absolute target larger than p0 is impossible,
    so absolute deltas are not simulated at low prevalence (Plan s6)."""
    rows = []
    for r in relative:
        for n in families:
            out = simulate(p0, -r * p0, n, m_per_arm, sigma)
            out["relative_reduction"] = r
            rows.append(out)
    return pd.DataFrame(rows)
