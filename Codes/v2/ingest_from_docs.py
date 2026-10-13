"""
ingest_from_docs.py - ingest human-annotated workbooks from Annotated_Docs
into the pipeline's canonical annotation/ directories.

Confidentiality: notes column is cleared before writing (institutional data).
Only label columns + unit_id are copied; no annotator identifiers are stored.

Usage (from Codes/v2):
    python ingest_from_docs.py           # ingest + reliability for all batches
    python ingest_from_docs.py --batch pilot   # single batch
    python ingest_from_docs.py --bank    # also process bank review verdicts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config_v2 as C
from annotation.schema import LABEL_FIELDS, primary_unnecessary, validate_labels, normalise_value
from annotation.sheets_io import ITEM_COLUMNS, LABEL_COLUMNS, write_sheet
import annotation.reliability as RL

DOCS_BASE = Path(r"D:\PhD\Attentional_Harms\Annotator_Packs\Annotated_Docs")
RETURNED_SRC = DOCS_BASE / "Returned"
ADJUD_SRC = DOCS_BASE / "_Adjudication"

# Map batch name -> list of source files (relative to RETURNED_SRC)
BATCH_FILES: dict[str, list[str]] = {
    "pilot":  ["Annotator_R1/Step1_Pilot_R1_checked.xlsx",
               "Annotator_R2/Step1_Pilot_R2_checked.xlsx",
               "Annotator_R3/Step1_Pilot_R3__checked.xlsx"],
    "main_1": ["Annotator_R1/Step2_Main_1_R1_checked.xlsx",
               "Annotator_R2/Step2_Main_1_R2_checked.xlsx",
               "Annotator_R3/Step2_Main_1_R3_checked.xlsx"],
    "main_2": ["Annotator_R1/Step3_Main_2_R1_checked.xlsx",
               "Annotator_R2/Step3_Main_2_R2_checked.xlsx",
               "Annotator_R3/Step3_Main_2_R3_checked.xlsx"],
    "main_3": ["Annotator_R1/Step4_Main_3_R1_checked.xlsx",
               "Annotator_R2/Step4_Main_3_R2_checked.xlsx",
               "Annotator_R3/Step4_Main_3_R3_checked.xlsx"],
    "rgr2":   ["Annotator_R1/Step5_RGR2_R1_checked.xlsx",
               "Annotator_R2/Step5_RGR2_R2_checked.xlsx",
               "Annotator_R3/Step5_RGR2_R3_checked.xlsx"],
    # Six published comparators on the 44 locked matched sets (Batch_comparators_R*.zip).
    # Names are matched by pattern because returned file names vary.
    "comparators": ["Annotator_R1/Step6_Comparators_R1*.xlsx",
                    "Annotator_R2/Step6_Comparators_R2*.xlsx",
                    "Annotator_R3/Step6_Comparators_R3*.xlsx"],
}

ADJUD_FILES: dict[str, str] = {
    "pilot":  "Step1_Pilot_adjudication_checked.xlsx",
    "main_1": "Step2_Main_1_adjudication_checked.xlsx",
    "main_2": "Step3_Main_2_adjudication_checked.xlsx",
    "main_3": "Step4_Main_3_adjudication_checked.xlsx",
    "rgr2":   "Step5_RGR2_adjudication_checked.xlsx",
    "comparators": "Step6_Comparators_adjudication*.xlsx",
}


def _resolve(base: Path, rel: str) -> Path:
    """Exact path, or the newest match when rel contains a wildcard."""
    if "*" not in rel:
        return base / rel
    hits = sorted((base / rel).parent.glob(Path(rel).name), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else base / rel.replace("*", "")


def _load_sheet(path: Path) -> pd.DataFrame:
    """Read first worksheet; strip rows with missing unit_id (or item_id for adjudication files)."""
    wb = load_workbook(path, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return pd.DataFrame()
    hdr = [str(h).strip() if h is not None else "" for h in rows[0]]
    df = pd.DataFrame(rows[1:], columns=hdr)
    # Adjudication files use item_id; normalise to unit_id
    if "item_id" in df.columns and "unit_id" not in df.columns:
        df = df.rename(columns={"item_id": "unit_id"})
    id_col = "unit_id" if "unit_id" in df.columns else None
    if id_col:
        df = df[df[id_col].notna() & (df[id_col].astype(str).str.strip() != "")]
    return df.reset_index(drop=True)


def _rater_code(fname: str) -> str:
    """Extract R1/R2/R3 study code from filename."""
    import re
    m = re.search(r"_(R\d)(?=[_.])", fname)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot extract rater code from {fname}")


def ingest_batch(batch: str) -> dict:
    """Copy annotation workbooks to annotation/returned/<batch>/, clear notes."""
    dest_dir = C.ANNOTATION_DIR / "returned" / batch
    dest_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for rel in BATCH_FILES[batch]:
        src = _resolve(RETURNED_SRC, rel)
        if not src.exists():
            print(f"  WARNING: not found: {src}", file=sys.stderr)
            continue
        rater = _rater_code(src.name)
        df = _load_sheet(src)
        # Clear notes (institutional free text — confidentiality constraint)
        if "notes" in df.columns:
            df["notes"] = ""
        # Validate that required label columns exist
        missing = [c for c in ITEM_COLUMNS + list(LABEL_FIELDS) if c not in df.columns]
        if missing:
            print(f"  WARNING {src.name}: missing columns {missing}", file=sys.stderr)
        dest = dest_dir / f"{batch}_{rater}.xlsx"
        write_sheet(df[[c for c in ITEM_COLUMNS if c in df.columns]].copy(), dest,
                    title=batch,
                    prefill=df[["unit_id"] + [c for c in LABEL_COLUMNS if c in df.columns]])
        counts[rater] = len(df)
        print(f"  wrote {len(df):3d} rows → {dest.name}")
    return counts


def ingest_adjudication(batch: str) -> int:
    """Read adjudicated labels file, save as annotation/reports/<batch>/adjudicated.csv."""
    src = _resolve(ADJUD_SRC, ADJUD_FILES[batch])
    if not src.exists():
        print(f"  WARNING: adjudication file not found: {src}", file=sys.stderr)
        return 0
    df = _load_sheet(src)
    # Rename item_id → unit_id if needed
    if "item_id" in df.columns and "unit_id" not in df.columns:
        df = df.rename(columns={"item_id": "unit_id"})

    label_cols = [c for c in LABEL_FIELDS if c in df.columns and c != "identity_spans"]
    keep = ["unit_id"] + label_cols
    out_df = df[keep].copy()

    # Compute primary outcome
    out_df["adjudicated_unnecessary"] = out_df["least_relevant"].map(
        lambda v: primary_unnecessary(str(v).strip().lower() if v is not None else None))

    # Normalise all label fields
    for f in label_cols:
        out_df[f] = out_df[f].map(lambda v, ff=f: normalise_value(ff, v))

    rep = C.ANNOTATION_DIR / "reports" / batch
    rep.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(rep / "adjudicated.csv", index=False, encoding="utf-8-sig")
    print(f"  adjudicated: {len(out_df)} rows → {rep / 'adjudicated.csv'}")
    return len(out_df)


def ingest_bank_review() -> dict:
    """Extract bank review verdicts (ok columns) without prompt text."""
    stats: dict = {}
    for rev in ("A", "B"):
        src = DOCS_BASE / "Returned" / f"Bank_Reviewer_{rev}" / f"Bank_Review_{rev}_checked.xlsx"
        if not src.exists():
            print(f"  WARNING: {src} not found", file=sys.stderr)
            continue
        wb = load_workbook(src, data_only=True)
        for sheet_name in ("Core", "Dialogue"):
            if sheet_name not in wb.sheetnames:
                continue
            ws = wb[sheet_name]
            rows = list(ws.iter_rows(values_only=True))
            hdr = [str(h).strip() if h is not None else "" for h in rows[0]]
            df = pd.DataFrame(rows[1:], columns=hdr)
            df = df[df["family_id"].notna()].reset_index(drop=True)
            # Keep only research columns: family_id, category, verdicts, proposed_edit
            # NOT: prompt text, identity_detail, identity_opening, neutral_opening, switch tasks
            verdict_cols = [c for c in ["family_id", "condition", "category",
                                        "category_ok", "relevance_ok", "natural_ok", "matched_ok",
                                        "proposed_edit"] if c in hdr]
            verdict_df = df[verdict_cols].copy()
            verdict_df["reviewer"] = rev
            key = f"{sheet_name.lower()}_{rev}"
            stats[key] = {"rows": len(verdict_df),
                          "all_ok": int((verdict_df[["category_ok", "relevance_ok",
                                                       "natural_ok", "matched_ok"]] == "yes").all(axis=1).sum()),
                          "with_edit": int(verdict_df["proposed_edit"].notna().sum())}
            dest_dir = C.DATA_DIR / "bank_review"
            dest_dir.mkdir(parents=True, exist_ok=True)
            verdict_df.to_csv(dest_dir / f"{key}_verdicts.csv", index=False, encoding="utf-8-sig")
            print(f"  bank review {key}: {len(verdict_df)} rows, {stats[key]['with_edit']} proposed edits")
    return stats


def run_reliability(batch: str) -> dict:
    """Run the existing reliability pipeline for a batch."""
    print(f"\n=== Reliability: {batch} ===")
    try:
        result = RL.run(batch)
        print(f"  alpha(primary)={result['primary_alpha']:.3f}, units={result['units']}, "
              f"problems={result['entry_problems']}")
        return result
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", help="only this batch (pilot|main_1|main_2|main_3|rgr2|comparators)")
    ap.add_argument("--bank", action="store_true", help="also process bank review verdicts")
    ap.add_argument("--skip-reliability", action="store_true")
    ap.add_argument("--refresh", action="store_true",
                    help="afterwards re-run the human-label analysis and regenerate the IPM tables")
    args = ap.parse_args()

    batches = [args.batch] if args.batch else list(BATCH_FILES)
    all_results: dict = {}

    for batch in batches:
        print(f"\n=== Ingesting {batch} ===")
        counts = ingest_batch(batch)
        n_adj = ingest_adjudication(batch)
        all_results[batch] = {"counts": counts, "adjudicated_units": n_adj}

        if not args.skip_reliability:
            rel = run_reliability(batch)
            all_results[batch]["reliability"] = rel

    if args.bank:
        print("\n=== Bank review ===")
        all_results["bank_review"] = ingest_bank_review()

    out_path = C.ANNOTATION_DIR / "reports" / "ingest_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\nSummary written to {out_path}")
    print(json.dumps(all_results, indent=2, default=str))

    if args.refresh:
        import subprocess
        here = Path(__file__).resolve().parent
        for cmd in (["analyze", "--labels", "human", "--split", "locked"], ["tables"]):
            subprocess.run([sys.executable, str(here / "run_all.py"), *cmd], cwd=here, check=True)


if __name__ == "__main__":
    main()
