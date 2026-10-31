"""
revision/method_cost_v2.py - deployment cost per final answer from the request logs (R11).

Separates (a) total deployed calls per answer, counting the shared B0 draft for methods that
post-process it, from (b) calls beyond an ordinary answer. Single-call methods REPLACE the ordinary
answer (they add a prompt, not a call). Tokens and latency come from the stored records.

    python revision/method_cost_v2.py  -> revision/results/method_cost_v2.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parent
G = pd.read_csv(REV / "data" / "gen_index.csv.gz", dtype=str)
for c in ("input_tokens", "output_tokens", "latency_ms"):
    G[c] = pd.to_numeric(G[c], errors="coerce").fillna(0)
G["draw_i"] = pd.to_numeric(G["draw"], errors="coerce")
K = ["family_id", "condition", "model_key", "draw_i"]

single = G[(G.role == "generation") & (G.experiment == "single") & (G.split == "locked") & (G.status == "ok")]
draft = single.groupby(K)[["input_tokens", "output_tokens", "latency_ms"]].first()
inter = G[(G.role == "generation") & (G.experiment == "intervention") & (G.split == "locked") & (G.status == "ok")]
steps = G[(G.role == "method_step") & (G.split == "locked") & (G.status == "ok")].copy()
steps["arm"] = steps["system_id"].str.split(":").str[0]
finals = G[(G.role == "final") & (G.split == "locked") & (G.status == "ok")]
REUSE = {"REPROMPT", "CAI_REVISE", "SYS2_IMPL", "RGR"}

rows = []
b0 = draft.reset_index()
rows.append({"arm": "B0", "answers": len(b0), "total_calls_mean": 1.0, "calls_beyond_ordinary_mean": 0.0,
             "tokens_mean": float((b0.input_tokens + b0.output_tokens).mean()),
             "latency_s_median": float(b0.latency_ms.median() / 1000)})
for a in ("B1", "B2"):
    s = inter[inter.arm == a]
    rows.append({"arm": a, "answers": len(s), "total_calls_mean": 1.0, "calls_beyond_ordinary_mean": 0.0,
                 "tokens_mean": float((s.input_tokens + s.output_tokens).mean()),
                 "latency_s_median": float(s.latency_ms.median() / 1000)})
for a in ["SI_NLI", "IF_MSC", "DAIQ_GUARD", "REPROMPT", "CAI_REVISE", "SYS2_IMPL", "RGR"]:
    f = finals[finals.arm == a][K + ["request_key"]]
    st = steps[steps.arm == a].groupby(K).agg(n=("request_key", "size"), it=("input_tokens", "sum"),
                                               ot=("output_tokens", "sum"), lat=("latency_ms", "sum"))
    m = f.join(st, on=K).fillna({"n": 0, "it": 0, "ot": 0, "lat": 0})
    if a in REUSE:
        m = m.join(draft, on=K)
        total_calls = m.n + 1
        tokens = m.it + m.ot + m.input_tokens + m.output_tokens
        lat = m.lat + m.latency_ms
    else:
        total_calls = m.n
        tokens = m.it + m.ot
        lat = m.lat
    rows.append({"arm": a, "answers": len(m), "total_calls_mean": float(total_calls.mean()),
                 "calls_beyond_ordinary_mean": float((total_calls - 1).mean()),
                 "tokens_mean": float(tokens.mean()), "latency_s_median": float(np.median(lat) / 1000)})
out = pd.DataFrame(rows)
out.to_csv(REV / "results" / "method_cost_v2.csv", index=False)
print(out.to_string(index=False))
