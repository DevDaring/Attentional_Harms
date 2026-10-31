"""
annotation/build_comparator_batch.py - human review batch for the six published comparators.

Uses the 44 locked matched sets of the intervention review: each comparator output is the
canonical 'final' record (experiment == 'comparators', role == 'final') for the same family,
condition, model and draw as an already-labelled B0 output. Raters therefore see the same
conversation and request as for B0/B1/B2/RGR; only the response differs, and the method is hidden.

    python -m annotation.build_comparator_batch      (from Codes/v2)
"""
from __future__ import annotations

import shutil
import zipfile

import pandas as pd

import config_v2 as C
import annotation.make_sheets as MS
from store import ResponseStore

BATCH = "comparators"
VERSION = "v2"
SEED = 20261012
COMPARATORS = ["SI_NLI", "IF_MSC", "DAIQ_GUARD", "REPROMPT", "CAI_REVISE", "SYS2_IMPL"]
KEYS = ["family_id", "condition", "model_key", "draw"]
PACKS = C.CODES_DIR.parent / "Annotator_Packs"


def matched_sets() -> pd.DataFrame:
    keys = pd.concat([pd.read_csv(MS.KEYS_DIR / f"{b}_key.csv", dtype=str) for b in ("main_1", "main_2", "main_3")])
    b0 = keys[(keys["sample"] == "intervention") & (keys["arm"] == "B0")].copy()
    b0["draw"] = b0["draw"].astype(float).astype(int)
    return b0[KEYS + ["cell", "category"]]


def build() -> list[str]:
    gen = MS.generation_frame(ResponseStore(C.OUTPUT_DIR / "generations.jsonl"))
    cmp_ = gen[(gen["experiment"] == "comparators") & gen["arm"].isin(COMPARATORS)].copy()
    cmp_["draw"] = cmp_["draw"].astype(int)
    sets = matched_sets()
    sample = cmp_.merge(sets.drop(columns="category"), on=KEYS, how="inner")
    counts = sample.groupby("arm").size()
    if len(sample) != len(sets) * len(COMPARATORS) or (counts != len(sets)).any():
        raise RuntimeError(f"Incomplete matched sets: {counts.to_dict()}")
    sample = sample.assign(sample="comparator_review", inclusion_prob=float("nan"))

    old_ids = set()
    old_key = MS.KEYS_DIR / f"{BATCH}_key.csv"
    if old_key.exists():
        old_ids = set(pd.read_csv(old_key, dtype=str)["unit_id"])
        shutil.copy(old_key, MS.KEYS_DIR / f"{BATCH}_key_v1_withdrawn.csv")
    out_dir = MS.write_batch(sample, BATCH, SEED)       # key file, registry, shuffled workbooks per rater
    new = pd.read_csv(old_key, dtype=str)
    if set(new["unit_id"]) & old_ids:
        raise RuntimeError("New unit ids collide with the withdrawn v1 batch")
    reg = pd.read_csv(MS.REGISTRY, dtype=str)       # drop withdrawn v1 ids from the registry
    reg[~reg["unit_id"].isin(old_ids)].to_csv(MS.REGISTRY, index=False)

    return package()


# Shared training material exactly as issued to the raters for Steps 1-5.
SHARED_SRC = PACKS / "Human_Labeling_v2" / "Annotator_R1"
SHARED_FILES = ["Annotation_Guidelines_v1.0.docx", "Quick_Reference_Card_v1.0.docx",
                "Practice_Workbook_v1.0.xlsx", "Practice_Answer_Key_v1.0.docx"]

README = """HUMAN LABELLING PACKAGE - Step 6 (rater {r})
Identity Information in Assistant Responses

This package is complete: everything you need for Step 6 is inside.

Files, in the order to use them:
  1. Annotation_Guidelines_v1.0.docx      Same guidelines as Steps 1-5. Re-read sections on
                                          relevance (Q6) and retention (Q9) before starting.
  2. Quick_Reference_Card_v1.0.docx       Keep it open while you label.
  3. Practice_Workbook_v1.0.xlsx          13 practice items for warm-up (optional if you finished
  4. Practice_Answer_Key_v1.0.docx        Steps 1-5 recently). Open the key only after practising.
  5. {sheet:<37}{n} items to label.
  6. Work_Log_{r}.xlsx                     One row per working session.

About Step 6
  Each item shows a conversation, the current request and one assistant response. Several items
  share the same request but have different responses, and some requests may look familiar from
  earlier steps. Label every item on its own, from the response in front of you. Do not look up
  your earlier answers or compare items with each other.

Workload
  {n} items at about 90 seconds each is roughly {hours:.1f} hours. Take a break about every
  45 minutes and do not label more than about 3 hours a day.

How to fill the workbook
  Use the drop-down list in each answer column (Q1-Q13). Do not edit the grey columns A-D, and do
  not add, delete or reorder rows. Keep the file name unchanged. Leave the Notes column empty
  unless an item is broken (for example, an empty or cut-off response).
  If you use Google Sheets, keep the file in one Google Sheet (version history on) and download it
  as .xlsx when you return it.

Work on your own. Do not discuss items with the other raters until all three have returned
their workbooks.

Return {sheet} and Work_Log_{r}.xlsx to the coordinator when finished.

Questions: author author, Department of Computer Science & Engineering, institution
E-mail: author-email
"""


def _work_log(rater: str, path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    ws = wb.active
    ws.title = "Work_Log"
    ws.append([f"Work log - annotator {rater} - Step 6"])
    ws.append(["date (dd-mm-yyyy)", "start time", "end time", "step / sheet", "items done", "comment"])
    ws["A1"].font = Font(bold=True)
    for c in ws[2]:
        c.font = Font(bold=True)
    for col, w in zip("ABCDEF", (18, 12, 12, 26, 12, 40)):
        ws.column_dimensions[col].width = w
    wb.save(path)


def package() -> list[str]:
    """One complete, self-contained zip per rater from the workbooks already in dist/."""
    import tempfile
    from pathlib import Path
    dist = MS.DIST_DIR / BATCH
    zips = []
    for rater in C.ANNOTATION["raters"]:
        src_sheet = dist / f"{BATCH}_{rater}.xlsx"
        n = len(pd.read_excel(src_sheet, sheet_name=0))
        sheet = f"Step6_Comparators_{rater}.xlsx"
        folder = f"Annotator_{rater}_Step6"
        target = PACKS / f"Annotator_{rater}_Step6_Comparators.zip"
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / f"Work_Log_{rater}.xlsx"
            _work_log(rater, log)
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr(f"{folder}/00_READ_ME_FIRST.txt",
                           README.format(r=rater, sheet=sheet, n=n, hours=n * C.ANNOTATION["seconds_per_unit"] / 3600))
                for name in SHARED_FILES:
                    z.write(SHARED_SRC / name, f"{folder}/{name}")
                z.write(src_sheet, f"{folder}/{sheet}")
                z.write(log, f"{folder}/Work_Log_{rater}.xlsx")
        zips.append(str(target))
    return zips


if __name__ == "__main__":
    import sys
    for z in (package() if "--package-only" in sys.argv else build()):
        print(z)
