"""
bank_review.py - two-person review of the prompt banks (Future plan s5.1).

  python bank_review.py export --reviewer A     -> data/review/Bank_Review_A.xlsx
  python bank_review.py import --reviewer A --file data/review/Bank_Review_A.xlsx

Each reviewer checks every row for category fit, task relevance of the identity
detail, naturalness and matching across conditions, and may propose an edit.
Import records the reviewer in reviewer_1 or reviewer_2. A row becomes
'reviewed_2' only when two different reviewers approved it with no unresolved
edit. Proposed edits are written to data/review/edits_<reviewer>.csv for the
authors to apply by hand; the importer never rewrites prompt text.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bank as B  # noqa: E402
import config_v2 as C  # noqa: E402

REVIEW_DIR = C.DATA_DIR / "review"
CHECKS = ["category_ok", "relevance_ok", "natural_ok", "matched_ok"]
CORE_SHOW = ["family_id", "category", "task_type", "condition", "prompt", "paraphrase", "identity_detail", "notes"]
DIA_SHOW = ["family_id", "category", "identity_opening", "neutral_opening", "switch_task_1", "switch_task_2",
            "switch_task_3", "identity_detail", "notes"]
GUIDE = [
    "category_ok: the identity detail belongs to the stated category (gender, caste, religion, race_ethnicity).",
    "relevance_ok: absent has no identity; incidental detail does not change what a good answer needs; "
    "required task really needs the detail and the prompt supplies every fact the answer depends on.",
    "natural_ok: a real user could plausibly write this, in respectful language.",
    "matched_ok: the three conditions (or the two dialogue openings) differ only as intended.",
    "Mark 'no' for any failed check and write the fix in proposed_edit. Leave proposed_edit empty if all are 'yes'.",
]


def _sheet(wb, title: str, df: pd.DataFrame, show: list[str]) -> None:
    ws = wb.create_sheet(title)
    cols = show + CHECKS + ["proposed_edit", "comment"]
    for ci, c in enumerate(cols, 1):
        cell = ws.cell(row=1, column=ci, value=c)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDEBF7")
        ws.column_dimensions[get_column_letter(ci)].width = 45 if c in (
            "prompt", "paraphrase", "identity_opening", "neutral_opening", "proposed_edit", "notes") else 16
    for ri, (_, r) in enumerate(df.iterrows(), 2):
        for ci, c in enumerate(show, 1):
            v = r.get(c, "")
            ws.cell(row=ri, column=ci, value="" if pd.isna(v) else str(v)).alignment = \
                Alignment(wrap_text=True, vertical="top")
    last = len(df) + 1
    dv = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
    for ci in range(len(show) + 1, len(show) + len(CHECKS) + 1):
        letter = get_column_letter(ci)
        dv.add(f"{letter}2:{letter}{last}")
    ws.add_data_validation(dv)
    ws.freeze_panes = "B2"


def export(reviewer: str) -> Path:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    bank = B.load_bank()
    bank = bank[bank["split"].isin(["dev", "locked"])]
    dia = B.load_dialogue()
    wb = Workbook()
    g = wb.active
    g.title = "Guide"
    for i, line in enumerate([f"Reviewer {reviewer}: prompt-bank review"] + GUIDE, 1):
        g.cell(row=i, column=1, value=line)
    g.column_dimensions["A"].width = 140
    _sheet(wb, "Core", bank, CORE_SHOW)
    _sheet(wb, "Dialogue", dia, DIA_SHOW)
    out = REVIEW_DIR / f"Bank_Review_{reviewer}.xlsx"
    wb.save(out)
    return out


def _read(path: Path, sheet: str) -> pd.DataFrame:
    ws = load_workbook(path, data_only=True)[sheet]
    rows = list(ws.iter_rows(values_only=True))
    return pd.DataFrame(rows[1:], columns=[str(h) for h in rows[0]]).fillna("")


def _status(row: pd.Series, reviewer: str, approved: bool) -> tuple[str, str, str]:
    r1, r2 = row["reviewer_1"], row["reviewer_2"]
    if not approved:
        return row["review_status"], r1, r2
    if not r1 or r1 == reviewer:
        return "reviewed_1", reviewer, r2
    if r1 != reviewer:
        return "reviewed_2", r1, reviewer
    return row["review_status"], r1, r2


def import_review(reviewer: str, path: Path) -> dict:
    bank, dia = B.load_bank(), B.load_dialogue()
    edits, counts = [], {"approved": 0, "flagged": 0, "blank": 0}
    for sheet, df, keys in (("Core", bank, ["family_id", "condition"]), ("Dialogue", dia, ["family_id"])):
        rev = _read(path, sheet)
        for _, r in rev.iterrows():
            vals = [str(r.get(c, "")).strip().lower() for c in CHECKS]
            if not any(vals):
                counts["blank"] += 1
                continue
            approved = all(v == "yes" for v in vals) and not str(r.get("proposed_edit", "")).strip()
            mask = pd.Series(True, index=df.index)
            for k in keys:
                mask &= df[k] == str(r[k])
            if not mask.any():
                continue
            i = df.index[mask][0]
            df.loc[i, ["review_status", "reviewer_1", "reviewer_2"]] = _status(df.loc[i], reviewer, approved)
            counts["approved" if approved else "flagged"] += 1
            if not approved:
                edits.append({"sheet": sheet, **{k: r[k] for k in keys}, "checks": ",".join(vals),
                              "proposed_edit": r.get("proposed_edit", ""), "comment": r.get("comment", "")})
    B.save_bank(bank)
    dia.to_csv(C.DIALOGUE_BANK, index=False, encoding="utf-8-sig")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(edits).to_csv(REVIEW_DIR / f"edits_{reviewer}.csv", index=False, encoding="utf-8-sig")
    counts["now_reviewed_2_core"] = int((bank["review_status"] == "reviewed_2").sum())
    counts["now_reviewed_2_dialogue"] = int((dia["review_status"] == "reviewed_2").sum())
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--reviewer", required=True)
    i = sub.add_parser("import"); i.add_argument("--reviewer", required=True); i.add_argument("--file", required=True)
    a = p.parse_args()
    if a.cmd == "export":
        print(export(a.reviewer))
    else:
        print(import_review(a.reviewer, Path(a.file)))


if __name__ == "__main__":
    main()
