"""
annotation/ingest.py - bring returned annotation and bank-review workbooks into the
pipeline WITHOUT any personal or institutional identifiers.

Confidentiality rules (requested by the authors' institutions):
  * Only cell values are copied; workbook metadata (author, last-modified-by, etc.)
    is dropped because new workbooks are written from scratch.
  * Institution codes of annotators and reviewers are replaced by study codes
    (R1-R3 for annotators, A/B for bank reviewers). The mapping is passed in at
    run time and never written to disk.
  * Adjudicator notes and AI-proposed rationales are not copied; only final
    decisions are kept.

Usage (from Codes/v2):
  python -m annotation.ingest pilot  --src <Annotated_Packs dir> --map AGT-01=R1,AGT-02=R2,AGT-04=R3
  python -m annotation.ingest review --src <reviewed dir>        --map AGT-16=A,AGT-06=B
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bank as B  # noqa: E402
import config_v2 as C  # noqa: E402
from annotation.schema import LABEL_FIELDS, primary_unnecessary  # noqa: E402
from annotation.sheets_io import ITEM_COLUMNS, LABEL_COLUMNS, write_sheet  # noqa: E402

CHECKS = ["category_ok", "relevance_ok", "natural_ok", "matched_ok"]


def _frame(path: Path, sheet: str | int = 0) -> pd.DataFrame:
    wb = load_workbook(path, data_only=True)
    ws = wb[sheet] if isinstance(sheet, str) else wb.worksheets[sheet]
    rows = list(ws.iter_rows(values_only=True))
    df = pd.DataFrame(rows[1:], columns=[str(h) for h in rows[0]])
    return df.dropna(how="all")


def _code_of(path: Path, mapping: dict[str, str]) -> str:
    for inst, study in mapping.items():
        if inst in path.as_posix():
            return study
    raise ValueError(f"No study code for {path.name}")


# ---------------------------------------------------------------- pilot / annotation batches
def ingest_batch(batch: str, src: Path, mapping: dict[str, str], ann_root: Path) -> dict:
    out = ann_root / "returned" / batch
    out.mkdir(parents=True, exist_ok=True)
    n = {}
    name_re = re.compile(rf"^(Batch|Step\d)_{re.escape(batch)}_R\d", re.IGNORECASE)
    for path in sorted(p for p in src.rglob("*.xlsx") if name_re.match(p.name)):
        if any(part.startswith("_") for part in path.parts):
            continue
        m = re.search(r"_(R\d)(?=[_.])", path.name)
        code = m.group(1) if m else _code_of(path, mapping)   # study code from the file name
        df = _frame(path, 0)
        missing = [c for c in ITEM_COLUMNS + list(LABEL_FIELDS) if c not in df.columns]
        if missing:
            raise ValueError(f"{code}: missing columns {missing}")
        # Research labels only: the free-text notes column is not copied (confidentiality).
        clean = df[[c for c in ITEM_COLUMNS + list(LABEL_FIELDS) if c in df.columns]].copy()
        clean["notes"] = ""
        write_sheet(clean[ITEM_COLUMNS], out / f"{batch}_{code}.xlsx", title=batch,
                    prefill=clean[["unit_id"] + [c for c in LABEL_COLUMNS if c in clean.columns]])
        n[code] = len(clean)
    adj_files = list(src.glob(f"_Adjudication/Batch_{batch.capitalize()}_adjudication*.xlsx"))
    if adj_files:
        fin = _frame(adj_files[0], "Final_labels").rename(columns={"item_id": "unit_id"})
        keep = ["unit_id"] + [f for f in LABEL_FIELDS if f in fin.columns and f != "identity_spans"]
        fin = fin[keep].copy()
        fin["adjudicated_unnecessary"] = fin["least_relevant"].map(
            lambda v: primary_unnecessary(str(v).strip().lower() if v is not None else None))
        rep = ann_root / "reports" / batch
        rep.mkdir(parents=True, exist_ok=True)
        fin.to_csv(rep / "adjudicated.csv", index=False, encoding="utf-8-sig")
        n["adjudicated_units"] = len(fin)
    return n


# ---------------------------------------------------------------- bank review
def _parse_dialogue_edit(text: str) -> dict:
    """'identity_opening: ... neutral_opening: ...' or plain text (= neutral opening)."""
    text = str(text).strip()
    parts = re.split(r"(identity_opening:|neutral_opening:)", text)
    if len(parts) == 1:
        return {"neutral_opening": text}
    out, key = {}, None
    for p in parts:
        if p in ("identity_opening:", "neutral_opening:"):
            key = p[:-1]
        elif key and p.strip():
            out[key] = p.strip()
    return out


def _agreement(a: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> dict:
    from analysis.agreement import krippendorff_alpha, raw_pairwise_agreement
    m = a.merge(b, on=keys, suffixes=("_A", "_B"))
    out = {"items": int(len(m))}
    for c in CHECKS:
        units = [[str(x).strip().lower(), str(y).strip().lower()] for x, y in zip(m[f"{c}_A"], m[f"{c}_B"])]
        out[c] = {"agreement": round(raw_pairwise_agreement(units), 4),
                  "alpha": round(krippendorff_alpha(units, "nominal", ["yes", "no"]), 4),
                  "no_A": int(sum(u[0] == "no" for u in units)), "no_B": int(sum(u[1] == "no" for u in units))}
    return out


def ingest_review(src: Path, mapping: dict[str, str]) -> dict:
    raw = {}
    for path in sorted(src.glob("*/Bank_Review_*.xlsx")):
        if path.parent.name.startswith("_"):
            continue
        raw[_code_of(path, mapping)] = {s: _frame(path, s) for s in ("Core", "Dialogue")}
    if set(raw) != {"A", "B"}:
        raise ValueError(f"Expected reviewers A and B, found {sorted(raw)}")
    stats = {"core": _agreement(raw["A"]["Core"], raw["B"]["Core"], ["family_id", "condition"]),
             "dialogue": _agreement(raw["A"]["Dialogue"], raw["B"]["Dialogue"], ["family_id"])}

    adj = src / "_Adjudication"
    core_fin = _frame(next(adj.glob("*Core_adjudication*.xlsx")), "Final_labels")
    dia_fin = _frame(next(adj.glob("*Dialogue_adjudication*.xlsx")), "Final_labels")
    stats["core"]["adjudicated_disagreements"] = int(len(_frame(next(adj.glob("*Core_adjudication*.xlsx")), "Disagreements")))
    stats["dialogue"]["adjudicated_disagreements"] = int(len(_frame(next(adj.glob("*Dialogue_adjudication*.xlsx")), "Disagreements")))

    bank, dia = B.load_bank(), B.load_dialogue()
    edits_log = []
    bank["_id"] = bank["family_id"] + "|" + bank["condition"]
    for _, r in core_fin[core_fin["proposed_edit"].notna()].iterrows():
        i = bank.index[bank["_id"] == r["item_id"]][0]
        edits_log.append({"item": r["item_id"], "field": "prompt", "old": bank.at[i, "prompt"],
                          "new": str(r["proposed_edit"]).strip()})
        bank.at[i, "prompt"] = str(r["proposed_edit"]).strip()
        bank.at[i, "adaptation_notes"] = "prompt edited after two-person review and adjudication (2026-10)"
    reviewed_core = set(core_fin["item_id"])
    mask = bank["_id"].isin(reviewed_core)
    bank.loc[mask, ["review_status", "reviewer_1", "reviewer_2"]] = ["reviewed_2", "A", "B"]
    bank = bank.drop(columns="_id")

    for _, r in dia_fin[dia_fin["proposed_edit"].notna()].iterrows():
        i = dia.index[dia["family_id"] == r["item_id"]][0]
        for field, new in _parse_dialogue_edit(r["proposed_edit"]).items():
            edits_log.append({"item": r["item_id"], "field": field, "old": dia.at[i, field], "new": new})
            dia.at[i, field] = new
    dia.loc[dia["family_id"].isin(set(dia_fin["item_id"])), ["review_status", "reviewer_1", "reviewer_2"]] = \
        ["reviewed_2", "A", "B"]

    B.save_bank(bank)
    dia.to_csv(C.DIALOGUE_BANK, index=False, encoding="utf-8-sig")
    review_dir = C.DATA_DIR / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(edits_log).to_csv(review_dir / "applied_edits.csv", index=False, encoding="utf-8-sig")
    (review_dir / "review_agreement.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    stats["edits_applied"] = len(edits_log)
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["pilot", "batch", "review"])
    p.add_argument("--src", required=True)
    p.add_argument("--batch", help="batch name for kind=batch, e.g. main_1")
    p.add_argument("--map", default="", help="INST=STUDY pairs (review only); kept in memory only")
    a = p.parse_args()
    mapping = dict(x.split("=") for x in a.map.split(",") if "=" in x)
    if a.kind in ("pilot", "batch"):
        name = "pilot" if a.kind == "pilot" else a.batch
        print(json.dumps(ingest_batch(name, Path(a.src), mapping, C.ANNOTATION_DIR), indent=2))
    else:
        print(json.dumps(ingest_review(Path(a.src), mapping), indent=2))


if __name__ == "__main__":
    main()
