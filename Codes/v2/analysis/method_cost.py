"""
analysis/method_cost.py - extra calls, tokens and cost per final answer for each
remedy (the B0 draft is shared and not counted as extra).
"""
from __future__ import annotations

import pandas as pd

from ledger import cost, load_prices, rates_for
import config_v2 as C


def method_cost(store) -> pd.DataFrame:
    prices = load_prices(C.PRICES_FILE)
    steps = [r for r in store.records("ok") if r.get("role") == "method_step"]
    finals = [r for r in store.records("ok") if r.get("role") == "final"]
    rows = []
    for r in steps:
        c = r["coords"]
        rows.append({"arm": c["arm"], "model_key": r["model_key"], "family_id": c["family_id"],
                     "condition": c["condition"], "draw": c["draw"], "in": r["input_tokens"],
                     "out": r["output_tokens"],
                     "usd": cost(rates_for(prices, r["model_key"])[1], r["input_tokens"], r["output_tokens"])})
    st = pd.DataFrame(rows)
    fin = pd.DataFrame([{"arm": f["coords"]["arm"], "model_key": f["model_key"],
                         "family_id": f["coords"]["family_id"], "condition": f["coords"]["condition"],
                         "draw": f["coords"]["draw"],
                         "gate": (f.get("trace") or {}).get("gate", "")} for f in finals])
    if fin.empty:
        return pd.DataFrame()
    keys = ["arm", "model_key", "family_id", "condition", "draw"]
    per = st.groupby(keys).agg(extra_calls=("in", "size"), extra_in=("in", "sum"),
                               extra_out=("out", "sum"), extra_usd=("usd", "sum")).reset_index() \
        if not st.empty else pd.DataFrame(columns=keys + ["extra_calls", "extra_in", "extra_out", "extra_usd"])
    m = fin.merge(per, on=keys, how="left").fillna({"extra_calls": 0, "extra_in": 0, "extra_out": 0,
                                                     "extra_usd": 0})
    out = m.groupby("arm").agg(finals=("draw", "size"), extra_calls_mean=("extra_calls", "mean"),
                               extra_in_mean=("extra_in", "mean"), extra_out_mean=("extra_out", "mean"),
                               usd_per_1000=("extra_usd", lambda x: 1000 * x.mean())).reset_index()
    gates = m[m["arm"] == "RGR"]["gate"].value_counts(normalize=True).to_dict()
    out.attrs["rgr_gate_outcomes"] = gates
    return out
