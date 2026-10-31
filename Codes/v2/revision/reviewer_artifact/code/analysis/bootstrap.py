"""
analysis/bootstrap.py - stratified family-cluster bootstrap (Future_Plan.md s7).

Whole families are resampled with replacement inside each category stratum,
so every variant, draw, model, arm and turn of a family moves together. A
statistic is a function stat(w) of a row-weight vector, where w[i] is the
number of times row i's family was drawn (times any fixed row weight).

Percentile intervals. The bootstrap p-value (2 x the smaller tail mass at 0)
is an approximation used only for Holm ordering of a small, pre-specified
contrast family.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

import config_v2 as C


class CellMean:
    """Equal-weight average over cells of the within-cell weighted mean of y,
    restricted to rows where mask is True and y is not NaN."""

    def __init__(self, df: pd.DataFrame, y: str, cells: list[str], mask=None,
                 base_weight: str | None = None):
        m = np.ones(len(df), dtype=bool) if mask is None else np.array(mask, dtype=bool, copy=True)
        yv = pd.to_numeric(df[y], errors="coerce").to_numpy(dtype=float)
        m &= ~np.isnan(yv)
        self.m = m
        self.y = np.where(m, yv, 0.0)
        codes = df[cells].astype(str).agg("|".join, axis=1) if cells else pd.Series(["all"] * len(df))
        self.codes, self.labels = pd.factorize(codes)
        self.k = len(self.labels)
        self.bw = np.ones(len(df)) if base_weight is None else \
            pd.to_numeric(df[base_weight], errors="coerce").fillna(0).to_numpy(dtype=float)

    def __call__(self, w: np.ndarray) -> float:
        ww = w * self.bw * self.m
        num = np.bincount(self.codes, weights=ww * self.y, minlength=self.k)
        den = np.bincount(self.codes, weights=ww, minlength=self.k)
        ok = den > 0
        if not ok.any():
            return float("nan")
        return float((num[ok] / den[ok]).mean())

    def n_rows(self) -> int:
        return int(self.m.sum())


class Bootstrap:
    def __init__(self, df: pd.DataFrame, family_col: str = "family_id",
                 strata_col: str = "category", reps: int = C.BOOTSTRAP_REPS, seed: int = 20261008):
        fams = df[family_col].astype(str).to_numpy()
        self.uniq, self.inv = np.unique(fams, return_inverse=True)
        fam_strata = df.groupby(df[family_col].astype(str))[strata_col].first().reindex(self.uniq)
        self.strata_idx = [np.where(fam_strata.to_numpy() == s)[0] for s in fam_strata.unique()]
        self.reps = reps
        rng = np.random.default_rng(seed)
        counts = np.zeros((reps, len(self.uniq)))
        for idx in self.strata_idx:
            draws = rng.integers(0, len(idx), size=(reps, len(idx)))
            for b in range(reps):
                np.add.at(counts[b], idx[draws[b]], 1)
        self.counts = counts                            # reps x families (row weights built lazily)
        self.n_rows = len(df)

    def run(self, stat: Callable[[np.ndarray], float]) -> dict:
        est = stat(np.ones(self.n_rows))
        boots = np.array([stat(self.counts[b][self.inv]) for b in range(self.reps)])
        fin = boots[np.isfinite(boots)]
        if len(fin) == 0 or not np.isfinite(est):
            return {"estimate": est, "ci_low": np.nan, "ci_high": np.nan, "se": np.nan,
                    "p_boot": np.nan, "n_families": len(self.uniq), "reps_used": 0}
        lo, hi = np.percentile(fin, [2.5, 97.5])
        p = min(1.0, 2 * min((fin <= 0).mean(), (fin >= 0).mean()))
        return {"estimate": est, "ci_low": float(lo), "ci_high": float(hi), "se": float(fin.std(ddof=1)),
                "p_boot": float(p), "n_families": len(self.uniq), "reps_used": int(len(fin))}

    def one_sided_lower(self, stat: Callable[[np.ndarray], float], level: float = 0.95) -> float:
        boots = np.array([stat(self.counts[b][self.inv]) for b in range(self.reps)])
        fin = boots[np.isfinite(boots)]
        return float(np.percentile(fin, 100 * (1 - level))) if len(fin) else float("nan")


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    items = sorted(((k, v) for k, v in pvals.items() if v == v), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = {"p": p, "p_holm": running, "reject": running <= alpha}
    return out
