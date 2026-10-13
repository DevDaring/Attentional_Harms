"""
analysis/secondary.py - secondary model-based sensitivity analysis
(Future_Plan.md s7: model-based analysis is secondary to the cluster bootstrap).

Logistic GEE with an exchangeable working correlation within family, model and
category as fixed factors. This is a population-averaged model, not the
logistic mixed model named in the plan; record it as such in protocol.md if it
is reported. Convergence and separation are checked and reported, never
hidden.
"""
from __future__ import annotations

import warnings

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def gee_logit(df: pd.DataFrame, outcome: str, rhs: str, group: str = "family_id") -> dict:
    d = df.dropna(subset=[outcome]).copy()
    if d[outcome].nunique() < 2:
        return {"ok": False, "reason": "outcome has a single value (separation)"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = smf.gee(f"{outcome} ~ {rhs}", groups=group, data=d,
                            family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
            fit = model.fit(maxiter=200)
        except Exception as exc:  # noqa: BLE001 - report every failure
            return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
    warn = [str(w.message) for w in caught]
    table = pd.DataFrame({"coef": fit.params, "se": fit.bse, "p": fit.pvalues})
    return {"ok": True, "n": int(len(d)), "groups": int(d[group].nunique()),
            "converged": bool(getattr(fit, "converged", True)), "warnings": warn,
            "table": table.reset_index().rename(columns={"index": "term"}).to_dict("records")}
