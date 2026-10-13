"""
annotation/build_pack.py - builds the annotator pack (zip) from
annotation/pack_src/content.py:

  00_READ_ME_FIRST.txt
  Annotation_Guidelines_v<V>.docx
  Quick_Reference_Card_v<V>.docx
  Practice_Workbook_v<V>.xlsx        (same layout as real batches, unlabelled)
  Practice_Answer_Key_v<V>.docx

A Markdown copy of the guidelines is written to annotation/guidelines.md for
version control. The build fails if practice labels violate schema.py.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Pt

import config_v2 as C
from annotation.pack_src import content as K
from annotation.schema import LABEL_FIELDS, validate_labels
from annotation.sheets_io import write_sheet

BUILD = C.ANNOTATION_DIR / "pack_src" / "build"


def _docx(blocks: list, path: Path) -> None:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    for kind, *rest in blocks:
        if kind == "h1":
            doc.add_heading(rest[0], level=1)
        elif kind == "h2":
            doc.add_heading(rest[0], level=2)
        elif kind == "p":
            doc.add_paragraph(rest[0])
        elif kind == "bullets":
            for item in rest[0]:
                doc.add_paragraph(item, style="List Bullet")
        elif kind == "numbered":
            for item in rest[0]:
                doc.add_paragraph(item, style="List Number")
        elif kind == "table":
            header, rows = rest
            t = doc.add_table(rows=1, cols=len(header))
            t.style = "Light Grid Accent 1"
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            for i, h in enumerate(header):
                t.rows[0].cells[i].text = h
            for row in rows:
                cells = t.add_row().cells
                for i, v in enumerate(row):
                    cells[i].text = str(v)
            doc.add_paragraph("")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)


def _markdown(blocks: list) -> str:
    out = []
    for kind, *rest in blocks:
        if kind == "h1":
            out.append(f"# {rest[0]}\n")
        elif kind == "h2":
            out.append(f"## {rest[0]}\n")
        elif kind == "p":
            out.append(rest[0] + "\n")
        elif kind == "bullets":
            out.append("\n".join(f"- {x}" for x in rest[0]) + "\n")
        elif kind == "numbered":
            out.append("\n".join(f"{i}. {x}" for i, x in enumerate(rest[0], 1)) + "\n")
        elif kind == "table":
            header, rows = rest
            out.append("| " + " | ".join(header) + " |")
            out.append("|" + "---|" * len(header))
            out += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
            out.append("")
    return "\n".join(out)


def _check_practice() -> None:
    for pid, _, _, _, labels, _ in K.PRACTICE:
        _, problems = validate_labels(labels)
        if problems:
            raise ValueError(f"Practice item {pid} violates the schema: {problems}")
        missing = set(LABEL_FIELDS) - set(labels)
        if missing:
            raise ValueError(f"Practice item {pid} lacks {missing}")


def _answer_key_blocks() -> list:
    blocks = [("h1", f"Practice Answer Key (version {K.VERSION})"),
              ("p", "Open this file only after you have completed the practice workbook on your own. "
                    "Compare each item with your answers and note any difference for the calibration call.")]
    for pid, before, request, response, labels, why in K.PRACTICE:
        blocks.append(("h2", f"{pid}"))
        if before != "(none)":
            blocks.append(("p", "Conversation before: " + before.replace("\n\n", " / ")))
        blocks.append(("p", "Current request: " + request))
        blocks.append(("p", "Response: " + response.replace("\n", " / ")))
        rows = [[f, str(labels[f]) if labels[f] != "" else "(empty)"] for f in LABEL_FIELDS]
        blocks.append(("table", ["Column", "Reference answer"], rows))
        blocks.append(("p", "Why: " + why))
    return blocks


def build(dest: Path | None = None) -> dict:
    _check_practice()
    v = K.VERSION
    BUILD.mkdir(parents=True, exist_ok=True)
    files = {
        "00_READ_ME_FIRST.txt": BUILD / "00_READ_ME_FIRST.txt",
        f"Annotation_Guidelines_v{v}.docx": BUILD / f"Annotation_Guidelines_v{v}.docx",
        f"Quick_Reference_Card_v{v}.docx": BUILD / f"Quick_Reference_Card_v{v}.docx",
        f"Practice_Workbook_v{v}.xlsx": BUILD / f"Practice_Workbook_v{v}.xlsx",
        f"Practice_Answer_Key_v{v}.docx": BUILD / f"Practice_Answer_Key_v{v}.docx",
    }
    files["00_READ_ME_FIRST.txt"].write_text(K.README, encoding="utf-8")
    _docx(K.GUIDELINES, files[f"Annotation_Guidelines_v{v}.docx"])
    _docx(K.QUICK_CARD, files[f"Quick_Reference_Card_v{v}.docx"])
    _docx(_answer_key_blocks(), files[f"Practice_Answer_Key_v{v}.docx"])
    items = pd.DataFrame([{"unit_id": pid, "conversation_before": b, "current_request": r, "response": s}
                          for pid, b, r, s, _, _ in K.PRACTICE])
    write_sheet(items, files[f"Practice_Workbook_v{v}.xlsx"], title="Practice")
    # Copy used by batch zips (make_sheets.write_batch looks for this name)
    _docx(K.GUIDELINES, BUILD / "Annotation_Guidelines.docx")
    (C.ANNOTATION_DIR / "guidelines.md").write_text(_markdown(K.GUIDELINES) + "\n\n" +
                                                    _markdown(K.QUICK_CARD), encoding="utf-8")
    dest = dest or (C.CODES_DIR.parent / "Annotator_Packs" / f"Annotator_Pack_v{v}_common.zip")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, src in files.items():
            z.write(src, f"Annotator_Pack_v{v}/{arc}")
    return {"zip": str(dest), "files": list(files), "practice_items": len(K.PRACTICE)}


if __name__ == "__main__":
    import json
    print(json.dumps(build(), indent=2))


def build_rater_zips(batch: str, dest_dir: Path | None = None) -> list[str]:
    """One zip per rater: the shared pack plus only that rater's workbook for `batch`."""
    import annotation.make_sheets as MS
    base = build()
    v = K.VERSION
    dest_dir = dest_dir or C.CODES_DIR.parent / "Annotator_Packs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for rater in C.ANNOTATION["raters"]:
        sheet = MS.DIST_DIR / batch / f"{batch}_{rater}.xlsx"
        if not sheet.exists():
            raise FileNotFoundError(sheet)
        target = dest_dir / (f"Annotator_Pack_v{v}_{rater}.zip" if batch == "pilot"
                             else f"Batch_{batch}_{rater}.zip")
        with zipfile.ZipFile(base["zip"]) as src, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
            if batch == "pilot":        # first delivery carries the whole pack
                for item in src.infolist():
                    z.writestr(item, src.read(item.filename))
            else:                       # later batches carry the guidelines and the quick card only
                for item in src.infolist():
                    if "Guidelines" in item.filename or "Quick_Reference" in item.filename:
                        z.writestr(item, src.read(item.filename))
            z.write(sheet, f"Annotator_Pack_v{v}/Batch_{batch.capitalize()}_{rater}.xlsx")
        out.append(str(target))
    return out
