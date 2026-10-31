"""
annotation/build_human_package.py - one master zip for the human labelling round (11 Oct 2026).

Layout of Human_Labeling_Package_v2.zip:
  00_COORDINATOR_GUIDE.txt                 for the coordinator only
  Annotator_R1/ R2/ R3/                    one folder per annotator (send only that folder)
      00_READ_ME_FIRST.txt, guidelines, quick card, practice workbook, answer key,
      Step1_Pilot_R#.xlsx, Step2_Main_1_R#.xlsx ... Step4_Main_3_R#.xlsx, Step5_RGR2_R#.xlsx,
      Work_Log_R#.xlsx
  Bank_Reviewer_A/ B/                      one folder per bank reviewer
      00_READ_ME_FIRST.txt, Bank_Review_<A|B>.xlsx (frozen bank), Work_Log_<A|B>.xlsx
Each folder is also written as its own zip next to the master zip.
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config_v2 as C  # noqa: E402
from annotation.build_pack import BUILD, build  # noqa: E402
from annotation.pack_src import content as K  # noqa: E402

DIST = C.ANNOTATION_DIR / "dist"
OUT = C.CODES_DIR.parent / "Annotator_Packs" / "Human_Labeling_v2"
STEPS = [("Step1_Pilot", "pilot", 60), ("Step2_Main_1", "main_1", 220), ("Step3_Main_2", "main_2", 220),
         ("Step4_Main_3", "main_3", 216), ("Step5_RGR2", "rgr2", 220)]

ANNOTATOR_README = """HUMAN LABELLING PACKAGE - Identity Information in Assistant Responses (rater {r})

Work through these files in order. Do each step on your own.

  1. Annotation_Guidelines_v1.0.docx      Read fully first (about 40 minutes).
  2. Quick_Reference_Card_v1.0.docx       Keep it open while you label.
  3. Practice_Workbook_v1.0.xlsx          13 practice items. Do them before opening the key.
  4. Practice_Answer_Key_v1.0.docx        Open only after finishing the practice workbook.
  5. Step1_Pilot_{r}.xlsx                  60 items. Return it before starting Step 2.
  6. Step2_Main_1_{r}.xlsx                 220 items.
  7. Step3_Main_2_{r}.xlsx                 220 items.
  8. Step4_Main_3_{r}.xlsx                 216 items.
  9. Step5_RGR2_{r}.xlsx                   220 items.
 10. Work_Log_{r}.xlsx                     One row per working session (date, start, end, step, items done).

In total there are 936 items; at about 90 seconds per item this is roughly 23 hours of work.
Take a break about every 45 minutes and do not label more than about 3 hours a day.

How to fill the workbooks: use the drop-down list in each answer column. Do not edit the grey
columns A-D, and do not add, delete or reorder rows. Keep each file name unchanged.
If you use Google Sheets, keep the file in one Google Sheet (version history on) and download it
as .xlsx when you return it.

Return each step's workbook and your work log to the coordinator when the step is finished.

Questions: {coord}
E-mail: {contact}
"""

REVIEWER_README = """PROMPT-BANK REVIEW (reviewer {x})

Bank_Review_{x}.xlsx has three sheets:
  Guide     - what each check means.
  Core      - 720 prompts (240 task families x 3 conditions).
  Dialogue  - 80 two-arm conversation openings with three follow-up tasks.

For every row, fill the four check columns with yes or no:
  category_ok, relevance_ok, natural_ok, matched_ok.
If any check is 'no', describe the problem or a better wording in proposed_edit. Use comment for
anything else. Work on your own; do not discuss rows with the other reviewer.

Expected time: about 6-8 hours. Record each session in Work_Log_{x}.xlsx.
Return the completed workbook and the work log to the coordinator.

Questions: {coord}
E-mail: {contact}
"""

COORDINATOR_GUIDE = """COORDINATOR GUIDE - human labelling round, version 2 (built 11 Oct 2026)

Send each person ONLY their own folder (or the matching per-person zip):
  Annotator_R1, Annotator_R2, Annotator_R3      three different people
  Bank_Reviewer_A, Bank_Reviewer_B                two different people (may not be R1-R3)

Order for annotators: practice -> answer key -> Step1 pilot (return it first) -> Steps 2-5.
The pilot labels are needed first: the judge-panel decision and the reliability screen use them.
Bank reviewers can work in parallel with the annotators.

Effort: 3 x ~23 h (annotators) + 2 x ~7 h (reviewers) = about 83 hours.

Returned files: keep every returned workbook exactly as the person sent it (no merging, no
re-saving, no scripts), together with the work log. Put them in
  Annotator_Packs/Human_Returns/<R1|R2|R3|A|B>/
Only the label columns are read for the study; identities stay with the coordinator.

Same items as before: these workbooks are the same samples (same item ids) as the batches issued
on 8-10 Oct 2026.
The bank-review workbooks show the frozen bank (after earlier edits); human review now decides
which families are excluded, not which prompts are changed.
"""


def _work_log(path: Path, who: str) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Work_Log"
    ws.append([f"Work log - {who}"])
    ws.append(["date (dd-mm-yyyy)", "start time", "end time", "step / sheet", "items done", "comment"])
    for c in ws[2]:
        c.font = Font(bold=True)
    for col, w in zip("ABCDEF", (18, 12, 12, 22, 12, 40)):
        ws.column_dimensions[col].width = w
    wb.save(path)


def main() -> None:
    import bank_review as BR
    build()                                         # guidelines, card, practice, key
    v = K.VERSION
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    folders = []
    for r in C.ANNOTATION["raters"]:
        d = OUT / f"Annotator_{r}"
        d.mkdir()
        (d / "00_READ_ME_FIRST.txt").write_text(
            ANNOTATOR_README.format(r=r, coord=K.COORDINATOR, contact=K.CONTACT), encoding="utf-8")
        for name in (f"Annotation_Guidelines_v{v}.docx", f"Quick_Reference_Card_v{v}.docx",
                     f"Practice_Workbook_v{v}.xlsx", f"Practice_Answer_Key_v{v}.docx"):
            shutil.copy2(BUILD / name, d / name)
        for step, batch, n in STEPS:
            src = DIST / batch / f"{batch}_{r}.xlsx"
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, d / f"{step}_{r}.xlsx")
        _work_log(d / f"Work_Log_{r}.xlsx", f"annotator {r}")
        folders.append(d)
    for x in ("A", "B"):
        d = OUT / f"Bank_Reviewer_{x}"
        d.mkdir()
        src = BR.export(x)                         # fresh export of the frozen bank
        shutil.copy2(src, d / f"Bank_Review_{x}.xlsx")
        (d / "00_READ_ME_FIRST.txt").write_text(
            REVIEWER_README.format(x=x, coord=K.COORDINATOR, contact=K.CONTACT), encoding="utf-8")
        _work_log(d / f"Work_Log_{x}.xlsx", f"bank reviewer {x}")
        folders.append(d)
    (OUT / "00_COORDINATOR_GUIDE.txt").write_text(COORDINATOR_GUIDE, encoding="utf-8")
    for d in folders:                              # per-person zips
        with zipfile.ZipFile(OUT / f"{d.name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(d.iterdir()):
                z.write(f, f"{d.name}/{f.name}")
    master = OUT.parent / "Human_Labeling_Package_v2.zip"
    with zipfile.ZipFile(master, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(OUT / "00_COORDINATOR_GUIDE.txt", "Human_Labeling_Package_v2/00_COORDINATOR_GUIDE.txt")
        for d in folders:
            for f in sorted(d.iterdir()):
                z.write(f, f"Human_Labeling_Package_v2/{d.name}/{f.name}")
    print(master)


if __name__ == "__main__":
    main()
