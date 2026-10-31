"""
annotation/make_sheets.py - draw annotation samples and write masked batches
(Future_Plan.md s6).

Samples
  pilot           60 units from pilot outputs (disjoint pilot families)
  representative  240 single-turn + 120 dialogue (first switched task) from
                  LOCKED families, balanced over model x category, condition /
                  history arm allocated evenly inside each cell; inclusion
                  probabilities recorded for weighting
  challenge       120 enriched diagnostic units (novel group labels, ambiguous
                  words, identity-required answers with mentions, attribution
                  phrasing); never used for prevalence
  intervention    60 matched B0/B1/B2 triplets from locked intervention-subset
                  families = 180 units, rated individually in random order

Every rater gets an independently shuffled copy. Model, arm, condition and
sample are hidden in a key file under annotation/keys/ that is never sent.
"""
from __future__ import annotations

import random
import re
import string
import zipfile
from pathlib import Path

import pandas as pd

import config_v2 as C
from annotation.sheets_io import write_sheet

KEYS_DIR = C.ANNOTATION_DIR / "keys"
DIST_DIR = C.ANNOTATION_DIR / "dist"
REGISTRY = KEYS_DIR / "unit_registry.csv"
_ATTRIBUTION_RE = re.compile(
    r"\b(as an?|since you are|you are (probably|likely)?\s*an?|being an?|given your)\s+"
    r"(woman|man|girl|boy|dalit|brahmin|muslim|hindu|christian|sikh|jain|buddhist|black|white|"
    r"asian|arab|female|male|mother|father|wife|husband|lower[- ]caste|upper[- ]caste)", re.I)


def generation_frame(store) -> pd.DataFrame:
    rows = []
    for r in store.records("ok"):
        if r.get("role") in ("judge", "method_step"):
            continue
        c = r.get("coords", {})
        rows.append({"request_key": r["request_key"], "experiment": r["experiment"],
                     "model_key": r["model_key"], "account": r["account"],
                     "family_id": c.get("family_id"), "condition": c.get("condition"),
                     "arm": c.get("arm"), "draw": c.get("draw"), "history_arm": c.get("history_arm"),
                     "trajectory": c.get("trajectory"), "turn": c.get("turn"),
                     "category": r.get("m_category"), "split": r.get("m_split"),
                     "messages": r["messages"], "text": r.get("text", "")})
    return pd.DataFrame(rows)


def _stratified(df: pd.DataFrame, cell_cols: list[str], sub_col: str | None, n_per_cell: int,
                rng: random.Random) -> pd.DataFrame:
    picks = []
    for ci, (cell, g) in enumerate(sorted(df.groupby(cell_cols), key=lambda kv: str(kv[0]))):
        if sub_col is None:
            subs = [(None, g)]
            alloc = [n_per_cell]
        else:
            subs = sorted(g.groupby(sub_col), key=lambda kv: str(kv[0]))
            base, extra = divmod(n_per_cell, len(subs))
            alloc = [base + (1 if (i + ci) % len(subs) < extra else 0) for i in range(len(subs))]
        for (sub, sg), k in zip(subs, alloc):
            idx = sorted(sg.index)
            chosen = rng.sample(idx, min(k, len(idx)))
            part = df.loc[chosen].copy()
            part["inclusion_prob"] = len(chosen) / len(idx) if idx else 0.0
            part["cell"] = "|".join(str(x) for x in (cell if isinstance(cell, tuple) else (cell,))) + \
                (f"|{sub}" if sub is not None else "")
            picks.append(part)
    return pd.concat(picks) if picks else df.iloc[0:0]


def sample_pilot(gen: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    pool = gen[gen["experiment"] == "pilot"]
    n = C.ANNOTATION["pilot_units"]
    cells = pool.groupby(["category", "condition"]).ngroups or 1
    out = _stratified(pool, ["category", "condition"], "model_key", max(1, n // cells), rng)
    return out.head(n).assign(sample="pilot")


def sample_representative(gen: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    single = gen[(gen["experiment"] == "single") & (gen["split"] == "locked") & (gen["arm"] == "B0")]
    n_cells = len(C.GENERATORS) * len(C.CATEGORIES)
    s = _stratified(single, ["model_key", "category"], "condition",
                    C.ANNOTATION["representative_single"] // n_cells, rng)
    dia = gen[(gen["experiment"] == "dialogue") & (gen["turn"] == 1)]
    d = _stratified(dia, ["model_key", "category"], "history_arm",
                    C.ANNOTATION["representative_dialogue"] // n_cells, rng)
    return pd.concat([s.assign(sample="representative_single"),
                      d.assign(sample="representative_dialogue")])


def sample_challenge(gen: pd.DataFrame, scores: pd.DataFrame, exclude: set, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    pool = gen[(gen["experiment"].isin(["single", "dialogue", "paraphrase"])) &
               (~gen["request_key"].isin(exclude))].merge(scores, on="request_key", how="left")
    pool = pool.fillna({"novel_group_label": 0, "n_ambiguous_unresolved": 0, "n_nationality": 0,
                        "n_group_label": 0, "n_role": 0, "n_identity_context": 0})
    alloc = C.CHALLENGE_ALLOCATION   # weighted towards likely positives (decision D6)
    buckets = {
        "rare_positive": pool[pool["novel_group_label"] == 1],
        "polysemy": pool[(pool["n_ambiguous_unresolved"] > 0) | (pool["n_nationality"] > 0)],
        "valid_mention": pool[(pool["condition"] == "required") &
                              ((pool["n_group_label"] + pool["n_role"] + pool["n_identity_context"]) > 0)],
        "attribution": pool[pool["text"].fillna("").str.contains(_ATTRIBUTION_RE)],
    }
    chosen, parts = set(), []
    for name, b in buckets.items():
        idx = [i for i in sorted(b.index) if pool.loc[i, "request_key"] not in chosen]
        take = rng.sample(idx, min(alloc.get(name, 0), len(idx)))
        chosen |= set(pool.loc[take, "request_key"])
        parts.append(pool.loc[take].assign(sample="challenge", cell=name))
    out = pd.concat(parts) if parts else pool.iloc[0:0]
    short = C.ANNOTATION["challenge_units"] - len(out)
    if short > 0:
        rest = [i for i in sorted(pool.index) if pool.loc[i, "request_key"] not in chosen]
        fill = rng.sample(rest, min(short, len(rest)))
        out = pd.concat([out, pool.loc[fill].assign(sample="challenge", cell="fill_random")])
    out["inclusion_prob"] = float("nan")       # enriched: not for prevalence
    return out[gen.columns.tolist() + ["sample", "cell", "inclusion_prob"]]


def _arm_frame(gen: pd.DataFrame, arm: str) -> pd.DataFrame:
    if arm == "B0":
        return gen[(gen["experiment"] == "single") & (gen["arm"] == "B0")]
    if arm in ("B1", "B2"):
        return gen[(gen["experiment"] == "intervention") & (gen["arm"] == arm)]
    return gen[(gen["experiment"] == "comparators") & (gen["arm"] == arm)]


def sample_intervention(gen: pd.DataFrame, seed: int, arms: list[str] | None = None,
                        split: str = "locked", sample_name: str = "intervention") -> pd.DataFrame:
    """Matched sets over C.INTERVENTION_REVIEW_ARMS (same family, condition, model, draw),
    locked families only, balanced over categories."""
    rng = random.Random(seed)
    keys = ["family_id", "condition", "model_key", "draw"]
    arms = arms or C.INTERVENTION_REVIEW_ARMS
    frames = {a: _arm_frame(gen[gen["split"] == split], a) for a in arms}
    index = {a: f.set_index(keys) for a, f in frames.items()}
    sets, per_cat = [], C.ANNOTATION["intervention_triplets"] // len(C.CATEGORIES)
    for cat in C.CATEGORIES:
        anchor = frames[arms[-1]]
        cands = [tuple(x) for x in anchor[anchor["category"] == cat][keys].drop_duplicates()
                 .itertuples(index=False)]
        rng.shuffle(cands)
        got = 0
        for k in cands:
            rows = [index[a].loc[[k]] if k in index[a].index else None for a in arms]
            if all(r is not None and len(r) == 1 for r in rows):
                sid = f"S{len(sets) // len(arms) + 1:03d}"
                for r in rows:
                    sets.append(r.reset_index().assign(sample=sample_name, cell=sid,
                                                       inclusion_prob=float("nan")))
                got += 1
            if got >= per_cat:
                break
    return pd.concat(sets, ignore_index=True) if sets else gen.iloc[0:0]


# ---------------------------------------------------------------- unit ids and sheets
def _registry() -> pd.DataFrame:
    if REGISTRY.exists():
        return pd.read_csv(REGISTRY, dtype=str)
    return pd.DataFrame(columns=["unit_id", "request_key", "batch"])


def assign_unit_ids(sample: pd.DataFrame, batch: str, seed: int) -> pd.DataFrame:
    reg = _registry()
    used = set(reg["unit_id"])
    rng = random.Random(f"{seed}-{batch}")
    ids = []
    for _ in range(len(sample)):
        while True:
            uid = "U" + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(6))
            if uid not in used:
                used.add(uid)
                ids.append(uid)
                break
    sample = sample.copy()
    sample["unit_id"] = ids
    new = pd.DataFrame({"unit_id": ids, "request_key": sample["request_key"].values, "batch": batch})
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([reg, new]).to_csv(REGISTRY, index=False)
    return sample


def _render_item(messages: list[dict]) -> tuple[str, str]:
    prior = messages[:-1]
    before = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in prior) if prior else "(none)"
    return before, messages[-1]["content"]


def write_batch(sample: pd.DataFrame, batch: str, seed: int) -> Path:
    """Write the key file and one shuffled workbook + zip per rater."""
    sample = assign_unit_ids(sample, batch, seed)
    items = []
    for _, r in sample.iterrows():
        before, current = _render_item(r["messages"])
        items.append({"unit_id": r["unit_id"], "conversation_before": before,
                      "current_request": current, "response": r["text"]})
    items = pd.DataFrame(items)
    key_cols = ["unit_id", "request_key", "sample", "cell", "inclusion_prob", "experiment",
                "model_key", "arm", "condition", "history_arm", "trajectory", "turn", "draw",
                "family_id", "category", "split"]
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    sample[[c for c in key_cols if c in sample.columns]].to_csv(KEYS_DIR / f"{batch}_key.csv",
                                                                  index=False)
    out_dir = DIST_DIR / batch
    out_dir.mkdir(parents=True, exist_ok=True)
    guide = C.ANNOTATION_DIR / "pack_src" / "build" / "Annotation_Guidelines.docx"
    for rater in C.ANNOTATION["raters"]:
        rng = random.Random(f"{seed}-{batch}-{rater}")
        order = list(items.index)
        rng.shuffle(order)
        xlsx = out_dir / f"{batch}_{rater}.xlsx"
        write_sheet(items.loc[order], xlsx, title=f"{batch}")
        with zipfile.ZipFile(out_dir / f"{batch}_{rater}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(xlsx, xlsx.name)
            if guide.exists():
                z.write(guide, guide.name)
    return out_dir
