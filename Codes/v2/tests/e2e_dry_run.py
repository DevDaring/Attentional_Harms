"""
tests/e2e_dry_run.py - end-to-end rehearsal on a synthetic full-size bank with
the mock provider and simulated annotators. Zero cost. Writes to
outputs_dryrun/e2e/. Run: python tests/e2e_dry_run.py
"""
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config_v2 as C  # noqa: E402
import run_all  # noqa: E402
from annotation.schema import LABEL_FIELDS  # noqa: E402
from annotation.sheets_io import read_sheet, write_sheet  # noqa: E402
from tests.synthetic import write_synthetic  # noqa: E402

OUT = C.DRYRUN_DIR / "e2e"


def rr(*args):
    base = ["--dry-run", "--out", str(OUT), "--bank", str(OUT / "prompt_bank.csv"),
            "--dialogue-bank", str(OUT / "dialogue_bank.csv"), "--stage", "reruns", "--workers", "8"]
    print("\n>>> run_all", " ".join(args))
    run_all.main(base + list(args))


def simulate_raters(batch: str, seed: int = 7) -> None:
    """Three raters who mostly agree: a shared latent label plus 12% noise."""
    dist = OUT / "annotation" / "dist" / batch
    ret = OUT / "annotation" / "returned" / batch
    ret.mkdir(parents=True, exist_ok=True)
    latent = {}
    for rater in C.ANNOTATION["raters"]:
        df = read_sheet(dist / f"{batch}_{rater}.xlsx")
        rng = random.Random(f"{seed}-{rater}")
        for i, r in df.iterrows():
            uid = r["unit_id"]
            if uid not in latent:
                lr = random.Random(uid)
                latent[uid] = lr.choice(["na", "na", "required", "useful_optional", "unnecessary"])
            rel = latent[uid] if rng.random() > 0.12 else rng.choice(["na", "unnecessary", "required"])
            mention = "no" if rel == "na" else "yes"
            labels = {"group_label_present": mention, "role_or_context_term_present": "no",
                      "person_gendered_pronoun": "no", "identity_spans": "woman" if mention == "yes" else "",
                      "least_supported": "na" if rel == "na" else "explicit", "least_relevant": rel,
                      "unsupported_personal_attribution": "no", "task_adequacy": rng.choice([1, 2, 2]),
                      "required_identity_retained": rng.choice(["na", "yes"]), "refusal": "no",
                      "stereotype": "no", "derogation": "no", "confidence": "high", "notes": ""}
            for k, v in labels.items():
                df.loc[i, k] = v
        write_sheet(df[["unit_id", "conversation_before", "current_request", "response"]],
                    ret / f"{batch}_{rater}.xlsx", prefill=df[["unit_id"] + list(LABEL_FIELDS) + ["notes"]])


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    write_synthetic(OUT)
    rr("assign")
    rr("check", "--mode", "freeze", "--splits", "dev,locked")
    rr("generate", "--experiment", "pilot")
    rr("gate", "--set", "protocol_frozen", "--note", "e2e rehearsal")
    for exp in ("single", "intervention", "paraphrase", "schema", "dialogue", "comparators"):
        rr("generate", "--experiment", exp, "--split", "all")
    rr("score")
    rr("sample", "--sample", "pilot", "--batch", "pilot")
    rr("sample", "--sample", "representative,challenge,intervention", "--batch", "main",
       "--max-per-batch", "220")
    for b in ("pilot", "main_1", "main_2", "main_3"):
        simulate_raters(b)
        rr("reliability", "--batch", b)
    rr("judge-calibrate")
    rr("audit")
    rr("gate", "--set", "evaluator_validated", "--note", "e2e rehearsal", "--override")
    rr("judge")
    rr("judge-third")
    rr("consensus")
    rr("analyze", "--labels", "judge", "--split", "locked")
    rr("analyze", "--labels", "human", "--split", "locked")
    rr("power")
    rr("method-cost")
    rr("cost")
    rr("status")
    print("\nE2E DRY RUN COMPLETE")


if __name__ == "__main__":
    main()
