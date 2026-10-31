"""
bank.py - prompt-bank schema, validation and deterministic split/subset
assignment (Future_Plan.md s5.1, s5.3).

prompt_bank.csv   one row per family x condition (absent / incidental / required)
dialogue_bank.csv one row per dialogue family

Validation modes
  draft  - structure only; problems are warnings (pilot and authoring)
  freeze - everything must pass: three conditions per family, design counts,
           two human reviews, licence/source, no duplicates, English only,
           no literal mask tokens, paraphrases where required
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import config_v2 as C

BANK_COLUMNS = ["family_id", "category", "task_type", "condition", "prompt", "paraphrase",
                "identity_detail", "split", "intervention_subset", "paraphrase_subset",
                "schema_subset", "source", "source_id", "license", "adaptation_notes",
                "provenance", "review_status", "reviewer_1", "reviewer_2", "notes"]
DIALOGUE_COLUMNS = ["family_id", "category", "split", "identity_opening", "neutral_opening",
                    "switch_task_1", "switch_task_2", "switch_task_3", "identity_detail",
                    "history_match_notes", "source", "source_id", "license", "provenance",
                    "review_status", "reviewer_1", "reviewer_2", "notes"]
SPLITS = {"pilot", "dev", "locked"}
REVIEWED = "reviewed_2"
_MASK_RE = re.compile(r"\bMASK\b|\[M\]|<mask>|\[MASK\]", re.IGNORECASE)
_NON_LATIN_RE = re.compile(r"[ऀ-෿؀-ۿ一-鿿Ѐ-ӿ]")


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, strict: bool, msg: str) -> None:
        (self.errors if strict else self.warnings).append(msg)

    def text(self) -> str:
        lines = [f"ERROR   {e}" for e in self.errors] + [f"warning {w}" for w in self.warnings]
        return "\n".join(lines) if lines else "no problems found"


def _bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def load_bank(path: Path = C.PROMPT_BANK) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for col in BANK_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    for col in ("intervention_subset", "paraphrase_subset", "schema_subset"):
        df[col] = df[col].map(_bool)
    return df[BANK_COLUMNS]


def load_dialogue(path: Path = C.DIALOGUE_BANK) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=DIALOGUE_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for col in DIALOGUE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[DIALOGUE_COLUMNS]


def save_bank(df: pd.DataFrame, path: Path = C.PROMPT_BANK) -> None:
    out = df.copy()
    for col in ("intervention_subset", "paraphrase_subset", "schema_subset"):
        out[col] = out[col].map(lambda b: "yes" if b else "")
    out.to_csv(path, index=False, encoding="utf-8-sig")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text.lower())).strip()


def _shingles(text: str, k: int = 3) -> set:
    toks = _norm(text).split()
    return {tuple(toks[i:i + k]) for i in range(max(1, len(toks) - k + 1))}


def _near_duplicates(texts: list[tuple[str, str]], threshold: float = 0.85) -> list[tuple[str, str, float]]:
    """(family_a, family_b, jaccard) for near-identical prompts in different families."""
    sh = [(fid, _shingles(t)) for fid, t in texts if t]
    found = []
    for a in range(len(sh)):
        for b in range(a + 1, len(sh)):
            if sh[a][0] == sh[b][0]:
                continue
            inter = len(sh[a][1] & sh[b][1])
            if not inter:
                continue
            jac = inter / len(sh[a][1] | sh[b][1])
            if jac >= threshold:
                found.append((sh[a][0], sh[b][0], round(jac, 3)))
    return found


def _text_checks(rep: Report, strict: bool, where: str, text: str) -> None:
    if _MASK_RE.search(text):
        rep.add(strict, f"{where}: literal mask token")
    if _NON_LATIN_RE.search(text):
        rep.add(strict, f"{where}: non-Latin script (English-only bank)")


def validate_bank(df: pd.DataFrame, mode: str = "draft", splits: set[str] | None = None) -> Report:
    strict = mode == "freeze"
    rep = Report()
    if splits:
        df = df[df["split"].isin(splits)]
    if df.empty:
        rep.add(True, "bank has no rows for the requested splits")
        return rep
    for col in ("family_id", "category", "condition", "prompt", "split"):
        blank = df[df[col].str.strip() == ""]
        if len(blank):
            rep.add(True, f"{len(blank)} rows with empty '{col}'")
    bad_cat = sorted(set(df["category"]) - set(C.CATEGORIES))
    if bad_cat:
        rep.add(True, f"unknown categories {bad_cat}")
    bad_cond = sorted(set(df["condition"]) - set(C.CONDITIONS))
    if bad_cond:
        rep.add(True, f"unknown conditions {bad_cond}")
    bad_split = sorted(set(df["split"]) - SPLITS)
    if bad_split:
        rep.add(True, f"unknown splits {bad_split}")
    dup_rows = df[df.duplicated(["family_id", "condition"], keep=False)]
    if len(dup_rows):
        rep.add(True, f"duplicate family/condition rows: {sorted(set(dup_rows['family_id']))[:10]}")
    for fid, g in df.groupby("family_id"):
        if set(g["condition"]) != set(C.CONDITIONS):
            rep.add(True, f"{fid}: conditions {sorted(set(g['condition']))} (need all three)")
        for col in ("category", "split", "task_type", "intervention_subset", "paraphrase_subset",
                    "schema_subset"):
            if g[col].nunique() > 1:
                rep.add(True, f"{fid}: '{col}' differs across conditions")
    exact = df[df.duplicated("prompt", keep=False) & (df["prompt"].str.strip() != "")]
    if len(exact):
        rep.add(True, f"{len(exact)} rows share an identical prompt text")
    for fa, fb, j in _near_duplicates(list(zip(df["family_id"], df["prompt"])))[:20]:
        rep.add(strict, f"near-duplicate prompts across families {fa} / {fb} (Jaccard {j})")
    for _, r in df.iterrows():
        where = f"{r['family_id']}/{r['condition']}"
        _text_checks(rep, True, where, r["prompt"])
        if r["paraphrase"]:
            _text_checks(rep, True, where + " paraphrase", r["paraphrase"])
        if r["paraphrase_subset"] and not r["paraphrase"].strip():
            rep.add(strict, f"{where}: in paraphrase subset but paraphrase empty")
        if r["condition"] == "required" and not r["identity_detail"].strip():
            rep.add(strict, f"{where}: required condition needs identity_detail (for retention)")
        if r["review_status"] != REVIEWED and r["split"] != "pilot":
            rep.add(strict, f"{where}: review_status '{r['review_status']}' (need {REVIEWED})")
        if not r["source"].strip() or not r["license"].strip():
            rep.add(strict, f"{where}: source/license missing")
    if strict:
        fam = df.drop_duplicates("family_id")
        expected = {"dev": C.DESIGN["dev_families"], "locked": C.DESIGN["locked_families"],
                    "pilot": C.DESIGN["pilot_families"]}
        for split, n in expected.items():
            if splits and split not in splits:
                continue
            got = (fam["split"] == split).sum()
            if got != n:
                rep.add(True, f"split {split}: {got} families (design {n})")
            per_cat = fam[fam["split"] == split]["category"].value_counts()
            if got and per_cat.nunique() > 1:
                rep.add(True, f"split {split}: unbalanced categories {per_cat.to_dict()}")
    return rep


def validate_dialogue(df: pd.DataFrame, mode: str = "draft") -> Report:
    strict = mode == "freeze"
    rep = Report()
    if df.empty:
        rep.add(strict, "dialogue bank is empty")
        return rep
    if df["family_id"].duplicated().any():
        rep.add(True, "duplicate dialogue family_id")
    for _, r in df.iterrows():
        fid = r["family_id"]
        for col in ("identity_opening", "neutral_opening", "switch_task_1", "switch_task_2",
                    "switch_task_3"):
            if not r[col].strip():
                rep.add(True, f"{fid}: '{col}' empty")
            else:
                _text_checks(rep, True, f"{fid}/{col}", r[col])
        if r["category"] not in C.CATEGORIES:
            rep.add(True, f"{fid}: unknown category {r['category']!r}")
        if r["review_status"] != REVIEWED:
            rep.add(strict, f"{fid}: review_status '{r['review_status']}' (need {REVIEWED})")
        a, b = len(r["identity_opening"].split()), len(r["neutral_opening"].split())
        if a and b and abs(a - b) / max(a, b) > 0.35:
            rep.add(False, f"{fid}: opening lengths differ by more than 35% ({a} vs {b} words)")
    if strict and len(df) != C.DESIGN["dialogue_families"]:
        rep.add(True, f"{len(df)} dialogue families (design {C.DESIGN['dialogue_families']})")
    return rep


def assign_splits_and_subsets(df: pd.DataFrame, seed: int = C.SUBSET_SEED,
                              overwrite: bool = False) -> pd.DataFrame:
    """Deterministic, category-balanced assignment made before any output is
    viewed. Families with split 'pilot' are left alone. Existing assignments are
    kept unless overwrite=True."""
    df = df.copy()
    rng = random.Random(seed)
    fam = df[df["split"] != "pilot"].drop_duplicates("family_id")[["family_id", "category", "split"]]
    split_of, inter, para, schema = {}, set(), set(), set()
    n_cat = len(C.CATEGORIES)
    for cat in C.CATEGORIES:
        fids = sorted(fam[fam["category"] == cat]["family_id"])
        rng.shuffle(fids)
        n_dev = C.DESIGN["dev_families"] // n_cat
        for i, fid in enumerate(fids):
            current = fam.loc[fam["family_id"] == fid, "split"].iloc[0]
            split_of[fid] = current if (current in ("dev", "locked") and not overwrite) else \
                ("dev" if i < n_dev else "locked")
        dev = [f for f in fids if split_of[f] == "dev"]
        locked = [f for f in fids if split_of[f] == "locked"]
        inter |= set(dev[: C.DESIGN["intervention_subset"]["dev"] // n_cat])
        inter |= set(locked[: C.DESIGN["intervention_subset"]["locked"] // n_cat])
        pool = sorted(fids)
        rng.shuffle(pool)
        para |= set(pool[: C.DESIGN["paraphrase_subset"] // n_cat])
        pool2 = sorted(fids)
        rng.shuffle(pool2)
        schema |= set(pool2[: C.DESIGN["schema_subset"] // n_cat])
    mask = df["split"] != "pilot"
    df.loc[mask, "split"] = df.loc[mask, "family_id"].map(split_of)
    if overwrite or not df.loc[mask, "intervention_subset"].any():
        df.loc[mask, "intervention_subset"] = df.loc[mask, "family_id"].isin(inter)
    if overwrite or not df.loc[mask, "paraphrase_subset"].any():
        df.loc[mask, "paraphrase_subset"] = df.loc[mask, "family_id"].isin(para)
    if overwrite or not df.loc[mask, "schema_subset"].any():
        df.loc[mask, "schema_subset"] = df.loc[mask, "family_id"].isin(schema)
    return df


def account_for_family(family_ids: list[str], categories: dict[str, str]) -> dict[str, str]:
    """Deterministic account allocation (Plan s8). 'primary': all to one account.
    'balanced': alternate within each category over sorted family ids."""
    aliases = list(C.ACCOUNTS)
    if C.ACCOUNT_MODE == "primary":
        return {f: C.PRIMARY_ACCOUNT for f in family_ids}
    out = {}
    for cat in sorted(set(categories.values())):
        fids = sorted(f for f in family_ids if categories[f] == cat)
        for i, f in enumerate(fids):
            out[f] = aliases[i % len(aliases)]
    return out
