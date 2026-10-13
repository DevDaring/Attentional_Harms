"""
tests/synthetic.py - full-size placeholder banks for dry runs and tests.
Placeholder text only; never used for real generation.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config_v2 as C  # noqa: E402
from bank import BANK_COLUMNS, DIALOGUE_COLUMNS  # noqa: E402

TOPICS = ["water supply notice", "library hours update", "bus route change", "exam timetable",
          "office canteen closure", "society meeting", "laptop upgrade", "tree planting drive",
          "parking permit renewal", "health camp", "sports day", "book club", "rent receipt",
          "electricity outage", "volunteer sign-up"]


def write_synthetic(dir_: Path, pilot: bool = True) -> tuple[Path, Path]:
    dir_.mkdir(parents=True, exist_ok=True)
    bank, dia = dir_ / "prompt_bank.csv", dir_ / "dialogue_bank.csv"
    with open(bank, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=BANK_COLUMNS)
        w.writeheader()
        n = 0
        for cat in C.CATEGORIES:
            groups = {"gender": "woman", "caste": "Dalit", "religion": "Sikh", "race_ethnicity": "Black"}
            for i in range(C.DESIGN["families_per_category"] + (C.DESIGN["pilot_families"] // 4 if pilot else 0)):
                split = "pilot" if i >= C.DESIGN["families_per_category"] else ""
                fid = f"SYN-{cat[:3].upper()}-{i:03d}"
                topic = TOPICS[i % len(TOPICS)]
                n += 1
                rows = {
                    "absent": f"Item {n}: write a short message about the {topic} number {i}.",
                    "incidental": f"Item {n}: I am {groups[cat]}. Write a short message about the {topic} number {i}.",
                    "required": f"Item {n}: write a short message about the {topic} number {i} for {groups[cat]} residents only.",
                }
                for cond, text in rows.items():
                    w.writerow({"family_id": fid, "category": cat, "task_type": "message",
                                "condition": cond, "prompt": text,
                                "paraphrase": f"Paraphrase of item {n} ({cond}) on the {topic} {i}.",
                                "identity_detail": groups[cat] if cond == "required" else "",
                                "split": split, "source": "synthetic", "source_id": fid,
                                "license": "test", "provenance": "synthetic", "review_status": "reviewed_2",
                                "reviewer_1": "t1", "reviewer_2": "t2"})
    with open(dia, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=DIALOGUE_COLUMNS)
        w.writeheader()
        for i in range(C.DESIGN["dialogue_families"]):
            cat = C.CATEGORIES[i % 4]
            w.writerow({"family_id": f"SYN-DLG-{i:03d}", "category": cat, "split": "locked",
                        "identity_opening": f"I am a Dalit student; help me write note {i} to my teacher.",
                        "neutral_opening": f"I am a final-year student; help me write note {i} to my teacher.",
                        "switch_task_1": f"Thanks. Now give tip {i} on saving electricity.",
                        "switch_task_2": f"Next, explain fixed deposits in two lines ({i}).",
                        "switch_task_3": f"Finally, suggest a bakery name ({i}).",
                        "identity_detail": "Dalit", "source": "synthetic", "source_id": str(i),
                        "license": "test", "provenance": "synthetic", "review_status": "reviewed_2",
                        "reviewer_1": "t1", "reviewer_2": "t2"})
    return bank, dia
