"""
revision/make_artifact.py - anonymised reviewer archive (review s12). Local only; NOT uploaded.

Copies data, prompts, protocol, freeze record, deviation log, labels, keys and scripts into
revision/reviewer_artifact/, scrubs author-identifying strings, refuses to finish if a secret-like
string is found, and writes reviewer_artifact.zip plus a SHA-256 manifest.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

V2 = Path(__file__).resolve().parent.parent
REV = V2 / "revision"
OUT = REV / "reviewer_artifact"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

FILES = {
    "protocol/protocol.md": V2 / "protocol.md",
    "protocol/gates.json": V2 / "manifests" / "gates.json",
    "protocol/deviations_post_freeze.md": V2 / "deviations_post_freeze.md",
    "protocol/prices.json": V2 / "manifests" / "prices.json",
    "data/prompt_bank.csv": V2 / "data" / "prompt_bank.csv",
    "data/dialogue_bank.csv": V2 / "data" / "dialogue_bank.csv",
    "annotation/guidelines.md": V2 / "annotation" / "guidelines.md",
    "annotation/schema.py": V2 / "annotation" / "schema.py",
    "labels/unit_ledger.csv": REV / "unit_ledger.csv",
    "labels/followup_FU1_key.csv": REV / "annotation_followup" / "keys" / "FU1_key.csv",
    "labels/followup_FU2_key.csv": REV / "annotation_followup" / "keys" / "FU2_key.csv",
    "labels/followup_FU1_raw_labels_long.csv": REV / "annotation_followup" / "reports" / "FU1" / "raw_labels_long.csv",
    "labels/followup_FU1_majority_labels.csv": REV / "annotation_followup" / "reports" / "FU1" / "majority_labels.csv",
    "labels/followup_FU1_reliability.json": REV / "annotation_followup" / "reports" / "FU1" / "reliability.json",
    "labels/followup_FU1_ingest_summary.json": REV / "annotation_followup" / "reports" / "FU1" / "ingest_summary.json",
    "labels/followup_FU2_raw_spans_long.csv": REV / "annotation_followup" / "reports" / "FU2" / "raw_spans_long.csv",
    "results/followup.json": REV / "results" / "followup.json",
    "results/followup_s1_pairs.csv": REV / "results" / "followup_s1_pairs.csv",
    "results/followup_s3_unnecessary.csv": REV / "results" / "followup_s3_unnecessary.csv",
    "results/followup_s3_retention.csv": REV / "results" / "followup_s3_retention.csv",
    "results/followup_s6_spans.csv": REV / "results" / "followup_s6_spans.csv",
    "results/followup_s6_units.csv": REV / "results" / "followup_s6_units.csv",
    "results/followup_cost.csv": REV / "results" / "followup_cost.csv",
    "outputs/labelled_texts.jsonl": REV / "data" / "labelled_texts.jsonl",
    "outputs/gen_index.csv.gz": REV / "data" / "gen_index.csv.gz",
    "results/reanalysis.json": REV / "results" / "reanalysis.json",
    "results/planning_sim.json": REV / "results" / "planning_sim.json",
    "results/method_cost_v2.csv": REV / "results" / "method_cost_v2.csv",
    "results/worked_examples.json": REV / "results" / "worked_examples.json",
    "results/human_locked_results.json": V2 / "outputs" / "analysis" / "human_locked" / "results.json",
    "results/robustness.json": V2 / "outputs" / "analysis" / "robustness" / "robustness.json",
    "results/evaluator_audit.json": V2 / "manifests" / "evaluator_audit.json",
    "results/evaluator_audit.units.csv": V2 / "manifests" / "evaluator_audit.units.csv",
    "plan/prospective_extension_plan.md": REV / "prospective_extension_plan.md",
    "plan/claim_evidence_matrix.csv": REV / "claim_evidence_matrix.csv",
}
CODE_DIRS = {"code/pipeline": [V2 / f for f in ("run_all.py", "config_v2.py", "remedies.py", "judges.py",
                                                "lexicon_v2.py", "scoring.py", "design.py", "bank.py",
                                                "executor.py", "store.py", "ledger.py", "gates.py", "common.py")],
             "code/analysis": sorted((V2 / "analysis").glob("*.py")),
             "code/annotation": sorted((V2 / "annotation").glob("*.py")),
             "code/providers": sorted((V2 / "providers").glob("*.py")),
             "code/revision": [p for p in sorted(REV.glob("*.py")) if not p.name.startswith("_") and p.name not in ("make_artifact.py", "make_blinded.py")]}
SCRUB = [(r"Abhinaba_Review\.md", "internal review notes"), (r"Abhinaba", "co-author"), (r"Basu", "co-author"),
         (r"Koushik", "author"), (r"\bDeb\b", "author"), (r"koushik[a-z0-9_.]*@[a-z.]+", "author-email"),
         (r"IIIT Kalyani|Kalyani", "institution"), (r"Allahabad|Prayagraj", "institution"),
         (r"Accenture", "employer"), (r"DevDaring", "repo-owner"), (r"github\.com/[^\s)\"']+", "repository"),
         (r"[A-Za-z]:\\\\?Users\\\\?[A-Za-z]+", "HOME"), (r"/c/Users/[A-Za-z]+", "HOME"), (r"\bDebz\b", "user")]
SECRET = re.compile(r"(AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_\-]{30,}|hf_[A-Za-z0-9]{30,})")


def scrub_text(s: str) -> str:
    for a, b in SCRUB:
        s = re.sub(a, b, s)
    return s


def put(rel: str, src: Path) -> None:
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".gz":
        with gzip.open(src, "rt", encoding="utf-8") as fh:
            txt = fh.read()
        txt = scrub_text(txt)
        if SECRET.search(txt):
            raise SystemExit(f"secret-like string in {src}")
        with gzip.open(dst, "wt", encoding="utf-8") as fh:
            fh.write(txt)
        return
    try:
        txt = src.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        shutil.copy2(src, dst)
        return
    txt = scrub_text(txt)
    if SECRET.search(txt):
        raise SystemExit(f"secret-like string in {src}")
    dst.write_text(txt, encoding="utf-8")


for rel, src in FILES.items():
    if src.exists():
        put(rel, src)
for d, files in CODE_DIRS.items():
    for f in files:
        if f.exists():
            put(f"{d}/{f.name}", f)

(OUT / "README.md").write_text("""# Reviewer archive (anonymised)

Contents
- `protocol/`: frozen protocol, freeze record with SHA-256 hashes (9 Oct 2026 11:43 UTC), post-freeze deviation log, price table.
- `data/`: prompt bank (240 families, three conditions, reviewer verdicts) and dialogue bank (80 families).
- `annotation/`: annotation guidelines and label schema.
- `labels/unit_ledger.csv`: one row per human-labelled response (1,200): sampling keys, inclusion probability, batch, the three raters' raw labels, majority, adjudication and final labels. Raters are coded R1-R3; free-text notes are excluded for annotator confidentiality.
- `outputs/labelled_texts.jsonl`: request messages and response text of every labelled response and of the drafts that remedies revised. `outputs/gen_index.csv.gz`: one row per stored generation record (no text): model, condition, arm, draw, tokens, latency, timestamps, RGR trace fields. The full 300 MB generation store is available from the editor on request.
- `results/`: all result files behind the manuscript tables.
- `code/`: the pipeline, analysis and revision scripts.

Regenerate every table (from a checkout with the full store):
```
python revision/build_ledger.py
python revision/reanalysis.py
python revision/method_cost_v2.py
python revision/make_tables.py
```
From this archive alone, `code/revision/reanalysis.py` sections that need only the ledger, texts and index run after copying `labels/`, `outputs/` into `revision/` and `revision/data/`.
""", encoding="utf-8")

manifest = {}
for f in sorted(OUT.rglob("*")):
    if f.is_file():
        manifest[str(f.relative_to(OUT)).replace("\\", "/")] = hashlib.sha256(f.read_bytes()).hexdigest()
(OUT / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
zp = REV / "reviewer_artifact.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            z.write(f, f"reviewer_artifact/{f.relative_to(OUT)}")
# final identity scan
bad = []
for f in OUT.rglob("*"):
    if f.is_file() and f.suffix in (".md", ".py", ".csv", ".json", ".jsonl"):
        t = f.read_text(encoding="utf-8", errors="ignore")
        for w in ("Abhinaba", "Basu", "Koushik", "Kalyani", "DevDaring", "koushikdeb"):
            if w in t:
                bad.append((str(f.relative_to(OUT)), w))
print("files", len(manifest), "zip MB %.1f" % (zp.stat().st_size / 1e6), "identity hits", bad[:10])
