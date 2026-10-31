"""
revision/build_ledger.py - Phase A evidence reconstruction (Abhinaba review, s6).

Outputs (all under revision/):
  data/gen_index.csv.gz      one row per stored generation record (no text): role, experiment,
                             arm, split, condition, family, model, draw, turn, status, tokens,
                             latency, timestamp, request_key, draft_key, RGR trace fields.
  data/labelled_texts.jsonl  request messages + response text for every human-labelled unit and
                             the B0 draft it was derived from (remedy arms).
  unit_ledger.csv            one row per human-labelled response (1,200): sampling keys,
                             inclusion probability, batch, linkage, generation metadata, each
                             rater's raw labels, majority labels, adjudication and final labels.
  denominators/*.csv         generation counts vs labelled units vs families.

Nothing here edits raw data. Run:  python revision/build_ledger.py
"""
from __future__ import annotations

import glob
import gzip
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REV = ROOT / "revision"
(REV / "data").mkdir(parents=True, exist_ok=True)
(REV / "denominators").mkdir(parents=True, exist_ok=True)

LABEL_FIELDS = ["group_label_present", "role_or_context_term_present", "person_gendered_pronoun",
                "identity_spans", "least_supported", "least_relevant", "unsupported_personal_attribution",
                "task_adequacy", "required_identity_retained", "refusal", "stereotype", "derogation",
                "confidence", "unnecessary"]


def load_keys() -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(str(ROOT / "annotation" / "keys" / "*_key.csv"))):
        if "withdrawn" in f:
            continue
        b = os.path.basename(f)[:-len("_key.csv")]
        k = pd.read_csv(f, dtype=str)
        k["batch"] = b
        parts.append(k)
    return pd.concat(parts, ignore_index=True)


def scan_generations(wanted: set[str]) -> tuple[pd.DataFrame, dict]:
    rows, texts = [], {}
    path = ROOT / "outputs" / "generations.jsonl"
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            co = r.get("coords") or {}
            tr = r.get("trace") or {}
            row = {
                "request_key": r.get("request_key"), "role": r.get("role"), "experiment": r.get("experiment"),
                "status": r.get("status"), "ts": r.get("ts"), "model_key": r.get("model_key"),
                "model_id": r.get("model_id"), "region": r.get("region"), "provider": r.get("provider"),
                "system_id": r.get("system_id"), "split": r.get("m_split"), "category": r.get("m_category"),
                "task_type": r.get("m_task_type"), "family_id": co.get("family_id"),
                "condition": co.get("condition"), "arm": co.get("arm"), "draw": co.get("draw"),
                "prompt_variant": co.get("prompt_variant"), "step": co.get("step"),
                "history_arm": co.get("history_arm"), "trajectory": co.get("trajectory"),
                "turn": r.get("turn") if r.get("turn") is not None else co.get("turn"),
                "temperature": (r.get("settings") or {}).get("temperature"),
                "max_tokens": (r.get("settings") or {}).get("max_tokens"),
                "input_tokens": r.get("input_tokens"), "output_tokens": r.get("output_tokens"),
                "latency_ms": r.get("latency_ms"), "stop_reason": r.get("stop_reason"),
                "had_reasoning": r.get("had_reasoning"), "n_attempts": len(r.get("attempts") or []),
                "error": (str(r.get("error"))[:120] if r.get("error") else None),
                "draft_key": tr.get("draft_key"), "rgr_gate": tr.get("gate"),
                "rgr_n_draft_terms": len(tr["draft_terms"]) if "draft_terms" in tr else None,
                "rgr_n_request_terms": len(tr["request_terms"]) if "request_terms" in tr else None,
                "rgr_remove": json.dumps(tr["remove"]) if "remove" in tr else None,
                "rgr_keep": json.dumps(tr["keep"]) if "keep" in tr else None,
                "rgr_restore": json.dumps(tr["restore"]) if "restore" in tr else None,
                "rgr_restore_suppressed": json.dumps(tr["restore_suppressed"]) if "restore_suppressed" in tr else None,
                "rgr_lost": json.dumps(tr["lost"]) if "lost" in tr else None,
                "trace_steps": json.dumps([s.get("step") for s in tr.get("steps", [])]) if tr else None,
                "n_words": len((r.get("text") or "").split()),
            }
            rows.append(row)
            if r.get("request_key") in wanted or tr.get("draft_key") in wanted:
                texts[r["request_key"]] = {"request_key": r["request_key"], "role": r.get("role"),
                                           "messages": r.get("messages"), "text": r.get("text"),
                                           "system": r.get("system"), "trace": tr or None,
                                           "coords": co, "model_key": r.get("model_key")}
    return pd.DataFrame(rows), texts


def main() -> None:
    keys = load_keys()
    wanted = set(keys["request_key"])
    gi, texts = scan_generations(wanted)
    # A generation record can be stored more than once (redo/exhausted); keep every row in the
    # index but flag duplicates of request_key.
    gi["dup_request_key"] = gi.duplicated("request_key", keep=False)
    gi.to_csv(REV / "data" / "gen_index.csv.gz", index=False, compression="gzip")

    # Add drafts of labelled remedy outputs to the text store.
    final = gi[gi["request_key"].isin(wanted)].drop_duplicates("request_key", keep="last").set_index("request_key")
    draft_keys = set(final["draft_key"].dropna())
    missing = draft_keys - set(texts)
    if missing:
        with open(ROOT / "outputs" / "generations.jsonl", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("request_key") in missing:
                    texts[r["request_key"]] = {"request_key": r["request_key"], "role": r.get("role"),
                                               "messages": r.get("messages"), "text": r.get("text"),
                                               "system": r.get("system"), "trace": r.get("trace"),
                                               "coords": r.get("coords"), "model_key": r.get("model_key")}
    with open(REV / "data" / "labelled_texts.jsonl", "w", encoding="utf-8") as fh:
        for t in texts.values():
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    # Raw labels, wide by rater.
    raws, majs, adjs = [], [], []
    for b in keys["batch"].unique():
        rep = ROOT / "annotation" / "reports" / b
        raw = pd.read_csv(rep / "raw_labels_long.csv", dtype=str)
        raws.append(raw)
        majs.append(pd.read_csv(rep / "majority_labels.csv", dtype=str))
        if (rep / "adjudicated.csv").exists():
            a = pd.read_csv(rep / "adjudicated.csv", dtype=str)
            a["adjudication_file_batch"] = b
            adjs.append(a)
    raw = pd.concat(raws, ignore_index=True)
    raw.columns = [c.lstrip("﻿") for c in raw.columns]
    wide = raw.pivot_table(index="unit_id", columns="rater", values=LABEL_FIELDS, aggfunc="first")
    wide.columns = [f"{r}_{f}" for f, r in wide.columns]
    wide = wide.reset_index()
    maj = pd.concat(majs, ignore_index=True)
    maj.columns = [c.lstrip("﻿") for c in maj.columns]
    adj = pd.concat(adjs, ignore_index=True) if adjs else pd.DataFrame(columns=["unit_id"])
    adj.columns = [c.lstrip("﻿") for c in adj.columns]
    adj = adj.add_prefix("adj_").rename(columns={"adj_unit_id": "unit_id"})

    L = keys.merge(wide, on="unit_id", how="left").merge(maj, on="unit_id", how="left") \
        .merge(adj, on="unit_id", how="left")
    meta_cols = ["request_key", "role", "status", "ts", "model_id", "region", "task_type", "prompt_variant",
                 "temperature", "max_tokens", "input_tokens", "output_tokens", "latency_ms", "stop_reason",
                 "had_reasoning", "n_attempts", "draft_key", "rgr_gate", "rgr_remove", "rgr_keep",
                 "rgr_restore", "rgr_restore_suppressed", "rgr_lost", "trace_steps", "n_words"]
    L = L.merge(final.reset_index()[meta_cols], on="request_key", how="left")
    L["adjudicated"] = L["adj_adjudicated_unnecessary"].notna()
    L["final_unnecessary"] = L["adj_adjudicated_unnecessary"].fillna(L["maj_unnecessary"])
    for f in ("unsupported_personal_attribution", "task_adequacy", "required_identity_retained", "refusal"):
        a = f"adj_{f}"
        L[f"final_{f}"] = L[a].fillna(L[f"maj_{f}"]) if a in L.columns else L[f"maj_{f}"]
    L["n_raters_unnecessary_yes"] = sum((L[f"R{i}_unnecessary"] == "yes").astype(int) for i in (1, 2, 3))
    L["text_hash"] = L["request_key"].map(
        lambda k: __import__("hashlib").sha256((texts.get(k, {}).get("text") or "").encode("utf-8")).hexdigest()[:16])
    L.to_csv(REV / "unit_ledger.csv", index=False, encoding="utf-8")

    # Denominator manifests -------------------------------------------------------------
    g = gi[gi["role"].isin(["generation", "final"]) & (gi["status"] == "ok")].drop_duplicates("request_key")
    gen_counts = g.groupby(["experiment", "arm", "split"]).agg(
        outputs=("request_key", "size"), families=("family_id", "nunique"),
        models=("model_key", "nunique")).reset_index()
    gen_counts.to_csv(REV / "denominators" / "generated_outputs.csv", index=False)
    lab = L.groupby(["sample", "experiment", "arm", "split"]).agg(
        units=("unit_id", "size"), families=("family_id", "nunique"), models=("model_key", "nunique"),
        positives_final=("final_unnecessary", lambda s: int((s == "yes").sum())),
        adjudicated=("adjudicated", "sum")).reset_index()
    lab.to_csv(REV / "denominators" / "labelled_units.csv", index=False)
    batch = L.groupby("batch").agg(units=("unit_id", "size"), families=("family_id", "nunique")).reset_index()
    batch["ratings"] = batch["units"] * 3
    batch.to_csv(REV / "denominators" / "annotation_batches.csv", index=False)
    print(gen_counts.to_string(index=False))
    print(lab.to_string(index=False))
    print(batch.to_string(index=False), "\nTOTAL units", int(batch.units.sum()), "ratings", int(batch.ratings.sum()))
    print("texts stored", len(texts), "missing drafts", len(draft_keys - set(texts)))


if __name__ == "__main__":
    main()
