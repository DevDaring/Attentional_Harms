"""
annotation/sheets_io.py - write and read annotator workbooks.

Columns A-D hold the item (read only); the label columns follow, with
drop-down lists generated from annotation/schema.py so raters can only
enter allowed values.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from annotation.schema import LABEL_FIELDS

ITEM_COLUMNS = ["unit_id", "conversation_before", "current_request", "response"]
LABEL_COLUMNS = list(LABEL_FIELDS) + ["notes"]
HEADER_TEXT = {
    "unit_id": "Item ID", "conversation_before": "Conversation before the current request",
    "current_request": "Current request", "response": "Response to rate", "notes": "Notes (optional)",
}
WIDTHS = {"unit_id": 11, "conversation_before": 45, "current_request": 45, "response": 60,
          "identity_spans": 24, "notes": 30}
_ITEM_FILL = PatternFill("solid", fgColor="F2F2F2")
_HEAD_FILL = PatternFill("solid", fgColor="DDEBF7")


def write_sheet(rows: pd.DataFrame, path: Path, title: str = "Items",
                prefill: pd.DataFrame | None = None) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = title
    cols = ITEM_COLUMNS + LABEL_COLUMNS
    for ci, col in enumerate(cols, 1):
        cell = ws.cell(row=1, column=ci, value=col)
        cell.font = Font(bold=True)
        cell.fill = _HEAD_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        note = HEADER_TEXT.get(col) or LABEL_FIELDS.get(col, (None, None, ""))[2]
        if note:
            cell.comment = Comment(note, "Guidelines")
        ws.column_dimensions[get_column_letter(ci)].width = WIDTHS.get(col, 16)
    for ri, (_, r) in enumerate(rows.iterrows(), 2):
        for ci, col in enumerate(ITEM_COLUMNS, 1):
            c = ws.cell(row=ri, column=ci, value=str(r.get(col, "") or ""))
            c.alignment = Alignment(wrap_text=True, vertical="top")
            c.fill = _ITEM_FILL
        if prefill is not None:
            pre = prefill.loc[prefill["unit_id"] == r["unit_id"]]
            if len(pre):
                for ci, col in enumerate(LABEL_COLUMNS, len(ITEM_COLUMNS) + 1):
                    val = pre.iloc[0].get(col, "")
                    ws.cell(row=ri, column=ci, value=None if val is None else val)
        for ci in range(len(ITEM_COLUMNS) + 1, len(cols) + 1):
            ws.cell(row=ri, column=ci).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[ri].height = 150
    last = max(2, len(rows) + 1)
    for ci, col in enumerate(LABEL_COLUMNS, len(ITEM_COLUMNS) + 1):
        allowed = LABEL_FIELDS.get(col, (None,))[0]
        if not allowed:
            continue
        dv = DataValidation(type="list", formula1='"' + ",".join(str(a) for a in allowed) + '"',
                            allow_blank=True, showErrorMessage=True,
                            errorTitle="Not an allowed value",
                            error="Choose a value from the drop-down list.")
        letter = get_column_letter(ci)
        dv.add(f"{letter}2:{letter}{last}")
        ws.add_data_validation(dv)
    ws.freeze_panes = "E2"
    qs = wb.create_sheet("Questions")
    qs.append(["Column", "Question", "Allowed values"])
    for f, (allowed, _, q) in LABEL_FIELDS.items():
        qs.append([f, q, ", ".join(str(a) for a in allowed) if allowed else "free text"])
    qs.column_dimensions["A"].width = 32
    qs.column_dimensions["B"].width = 95
    qs.column_dimensions["C"].width = 55
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def read_sheet(path: Path) -> pd.DataFrame:
    wb = load_workbook(path, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return pd.DataFrame()
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    df = pd.DataFrame(rows[1:], columns=header)
    df = df[df["unit_id"].notna() & (df["unit_id"].astype(str).str.strip() != "")]
    return df
