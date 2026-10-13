"""
analysis/agreement.py - inter-rater agreement statistics.

Krippendorff's alpha (nominal, ordinal, interval) via the coincidence matrix:
  Krippendorff, K. (2011). Computing Krippendorff's Alpha-Reliability.
  Annenberg School for Communication, University of Pennsylvania.
Gwet's AC1 for several raters:
  Gwet, K. L. (2008). Computing inter-rater reliability and its variance in the
  presence of high agreement. Br. J. Math. Stat. Psychol., 61(1), 29-48.

Plan s6: report alpha, raw agreement, prevalence and class-specific confusion;
AC1 is a sensitivity check only.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np


def _clean(units: list[list]) -> list[list]:
    return [[v for v in u if v is not None and v == v and v != ""] for u in units]


def krippendorff_alpha(units: list[list], level: str = "nominal", categories: list | None = None) -> float:
    """units: one list of rater values per unit (missing = None). Units with
    fewer than two values are ignored. Returns nan when undefined."""
    units = [u for u in _clean(units) if len(u) >= 2]
    if not units:
        return float("nan")
    values = sorted({v for u in units for v in u}, key=lambda x: (categories.index(x)
                                                                   if categories and x in categories
                                                                   else str(x)))
    if categories:
        values = [c for c in categories if c in values] + [v for v in values if v not in categories]
    idx = {v: i for i, v in enumerate(values)}
    k = len(values)
    o = np.zeros((k, k))
    for u in units:
        m = len(u)
        for a, b in ((x, y) for i, x in enumerate(u) for j, y in enumerate(u) if i != j):
            o[idx[a], idx[b]] += 1.0 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n <= 1 or k < 2:
        return 1.0 if k == 1 else float("nan")
    delta = np.zeros((k, k))
    for c in range(k):
        for e in range(k):
            if level == "nominal":
                delta[c, e] = 0.0 if c == e else 1.0
            elif level == "ordinal":
                lo, hi = min(c, e), max(c, e)
                delta[c, e] = (n_c[lo:hi + 1].sum() - (n_c[c] + n_c[e]) / 2.0) ** 2
            elif level == "interval":
                delta[c, e] = (float(values[c]) - float(values[e])) ** 2
            else:
                raise ValueError(level)
    d_o = (o * delta).sum() / n
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    return float("nan") if d_e == 0 else float(1.0 - d_o / d_e)


def gwet_ac1(units: list[list], categories: list) -> float:
    units = [u for u in _clean(units) if len(u) >= 2]
    if not units or len(categories) < 2:
        return float("nan")
    q = len(categories)
    pa_terms, pi = [], np.zeros(q)
    for u in units:
        r = len(u)
        cnt = Counter(u)
        pa_terms.append(sum(c * (c - 1) for c in cnt.values()) / (r * (r - 1)))
        for i, cat in enumerate(categories):
            pi[i] += cnt.get(cat, 0) / r
    pa = float(np.mean(pa_terms))
    pi /= len(units)
    pe = float((pi * (1 - pi)).sum() / (q - 1))
    return float("nan") if pe == 1 else (pa - pe) / (1 - pe)


def raw_pairwise_agreement(units: list[list]) -> float:
    agree = total = 0
    for u in _clean(units):
        for a, b in combinations(u, 2):
            total += 1
            agree += int(a == b)
    return agree / total if total else float("nan")


def prevalence(units: list[list]) -> dict:
    cnt = Counter(v for u in _clean(units) for v in u)
    total = sum(cnt.values())
    return {str(k): round(v / total, 4) for k, v in sorted(cnt.items(), key=lambda kv: str(kv[0]))}


def pairwise_confusion(rater_a: list, rater_b: list, categories: list) -> dict:
    table = {str(a): {str(b): 0 for b in categories} for a in categories}
    for a, b in zip(rater_a, rater_b):
        if a in categories and b in categories:
            table[str(a)][str(b)] += 1
    return table


def summarise_field(units: list[list], categories: list, level: str) -> dict:
    return {"n_units": sum(1 for u in _clean(units) if len(u) >= 2),
            "alpha": krippendorff_alpha(units, level, categories),
            "alpha_level": level,
            "raw_agreement": raw_pairwise_agreement(units),
            "gwet_ac1": gwet_ac1(units, categories) if level == "nominal" else float("nan"),
            "prevalence": prevalence(units)}
