"""Unit tests for the v2 pipeline. Run: python -m pytest tests -q (from Codes/v2)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bank as B  # noqa: E402
import config_v2 as C  # noqa: E402
import design as D  # noqa: E402
import judges as J  # noqa: E402
import lexicon_v2 as L  # noqa: E402
from analysis.agreement import gwet_ac1, krippendorff_alpha  # noqa: E402
from analysis.bootstrap import Bootstrap, CellMean, holm  # noqa: E402
from annotation.schema import primary_unnecessary, validate_labels  # noqa: E402
from common import extract_json_object, strip_reasoning  # noqa: E402
from executor import Runner  # noqa: E402
from ledger import BudgetExceeded, Ledger, load_prices  # noqa: E402
from providers.mock import MockProvider  # noqa: E402
from scoring import lexical_features, legacy_gss  # noqa: E402
from store import ResponseStore  # noqa: E402
from tests.synthetic import write_synthetic  # noqa: E402


# ------------------------------------------------------------- lexicon
def _layers(text):
    return [(s["canonical"], s["layer"], s["resolved"]) for s in L.find_spans(text)]


def test_he_inside_the_is_not_matched():
    assert L.find_spans("the theme of the thesis") == []


def test_generic_they_is_pronoun_only():
    s = L.summarise("They should save the file.")
    assert s["n_pronoun"] == 1 and s["n_group_label"] == 0 and s["density_non_pronoun"] == 0


def test_colours_are_not_race():
    spans = L.find_spans("Paint the walls white and add a black shelf.")
    assert all(not s["resolved"] for s in spans)


def test_race_resolved_with_person_noun():
    assert ("black", "group_label", True) in _layers("Black students met white families.")


def test_multiword_longest_match_and_no_overlap():
    spans = L.find_spans("Scheduled Caste students and an African American woman")
    canon = [s["canonical"] for s in spans]
    assert "scheduled caste" in canon and "caste" not in canon
    assert "african american" in canon and "african" not in canon and "american" not in canon


def test_plural_and_possessive_aliases():
    assert {"dalit", "muslim", "woman"} <= L.group_label_set("Dalits, Muslims and women's groups")


def test_miss_verb_vs_title():
    lay = _layers("Miss Rao said you might miss the bus.")
    assert ("miss", "role", True) in lay and ("miss", "role", False) in lay


def test_religious_required_terms_are_context():
    s = L.summarise("Eid Mubarak! Wishing you joy this Eid al-Fitr.")
    assert s["n_identity_context"] == 2


def test_history_relative_novelty():
    f = lexical_features("As a Sikh student, I request a receipt.", ["I'm a Sikh student.", "Rent receipt?"])
    assert f["any_group_label"] == 1 and f["novel_group_label"] == 0
    f2 = lexical_features("As a Hindu, you value cleanliness.", ["Fix the tap please."])
    assert f2["novel_group_label"] == 1


def test_legacy_gss_matches_may_2026_definition():
    assert legacy_gss("she is a woman") == pytest.approx(2 / 4)


# ------------------------------------------------------------- helpers
def test_json_extraction_variants():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('Sure! {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}
    assert extract_json_object("no json") is None


def test_strip_reasoning():
    assert strip_reasoning("<think>x</think> Final") == ("Final", True)
    assert strip_reasoning("<think>unfinished") == ("", True)


def test_schema_primary_and_consistency():
    assert primary_unnecessary("unnecessary") == "yes"
    assert primary_unnecessary("required") == "no"
    assert primary_unnecessary("unclear") == "unclear"
    _, probs = validate_labels({"group_label_present": "no", "role_or_context_term_present": "no",
                                "person_gendered_pronoun": "no", "least_supported": "explicit",
                                "least_relevant": "required", "unsupported_personal_attribution": "no",
                                "task_adequacy": 2, "required_identity_retained": "na", "refusal": "no",
                                "stereotype": "no", "derogation": "no"})
    assert any("should be 'na'" in p for p in probs)


# ------------------------------------------------------------- agreement
def test_krippendorff_reference_values():
    N = None
    A = [1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N]
    Bb = [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3]
    Cc = [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N]
    Dd = [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N]
    units = [[A[i], Bb[i], Cc[i], Dd[i]] for i in range(12)]
    assert krippendorff_alpha(units, "nominal") == pytest.approx(0.743, abs=1e-3)
    assert krippendorff_alpha(units, "interval") == pytest.approx(0.849, abs=1e-3)
    assert krippendorff_alpha(units, "ordinal", [1, 2, 3, 4, 5]) == pytest.approx(0.815, abs=1e-3)


def test_ac1_perfect_agreement():
    assert gwet_ac1([["yes", "yes", "yes"], ["no", "no", "no"]], ["yes", "no"]) == pytest.approx(1.0)


# ------------------------------------------------------------- design and bank
def test_design_counts_match_plan():
    c = D.design_call_counts()
    core = {k: v for k, v in c.items() if k != "comparators"}
    assert core == {"pilot": 288, "single": 8640, "dialogue": 7680, "intervention": 8640,
                    "paraphrase": 2160, "schema": 720}
    assert sum(core.values()) == 28128          # original plan, unchanged
    cc = D.comparator_call_counts()
    # 6 comparators on 80 locked families, RGR on 120 families; 36 outputs per family
    assert cc["finals"] == 6 * 80 * 36 + 120 * 36
    assert cc["single_calls"] == 3 * 80 * 36
    assert cc["multi_calls"] == (1 + 2 + 2) * 80 * 36 + 2 * 120 * 36
    j = D.judge_call_counts(c)
    assert j["judged_outputs"] == 27120 + cc["finals"]
    assert j["calibration"] == 720 * 4


def test_synthetic_bank_units_match_design(tmp_path):
    bank_p, dia_p = write_synthetic(tmp_path)
    bank = B.assign_splits_and_subsets(B.load_bank(bank_p))
    assert B.validate_bank(bank, "freeze").ok, B.validate_bank(bank, "freeze").text()
    dia = B.load_dialogue(dia_p)
    # rgr_ro is post hoc (10 Oct 2026) and outside the frozen design counts
    counts = {e: sum(u.n_calls for u in D.build_units(e, bank, dia)) for e in D.EXPERIMENTS if e != "rgr_ro"}
    assert counts == D.design_call_counts()


def test_validator_catches_mask_devanagari_and_duplicates(tmp_path):
    bank_p, _ = write_synthetic(tmp_path)
    bank = B.load_bank(bank_p)
    bank.loc[0, "prompt"] = "MASK महिलाएँ"
    bank.loc[3, "prompt"] = bank.loc[4, "prompt"]
    rep = B.validate_bank(bank, "draft")
    text = rep.text()
    assert "mask" in text and "non-Latin" in text and "identical prompt" in text


def test_judge_assignment_avoids_own_family():
    for g in C.GENERATORS:
        js = J.assign_judges(g)
        assert len(set(js)) == 2
        assert all(C.MODELS[j]["family"] != C.MODELS[g]["family"] for j in js)
        third = J.third_judge(g, js)
        assert third is None or C.MODELS[third]["family"] not in {C.MODELS[x]["family"] for x in js + [g]}


def test_parse_judgment_abstains_on_garbage():
    assert J.parse_judgment("not json")["unnecessary"] == "abstain"


# ------------------------------------------------------------- ledger and executor
def _ledger(tmp_path, stage="access_pilot"):
    return Ledger(tmp_path / "ledger.jsonl", load_prices(C.PRICES_FILE), stage, C.STAGE_CEILINGS,
                  C.HARD_CAP_USD, C.ACCOUNT_CEILINGS)


def test_ledger_ceiling_and_orphan_replay(tmp_path):
    led = _ledger(tmp_path)
    rid = led.reserve("nova_lite", "aws_a", 1000, 400, "t")
    led.settle(rid, 900, 300)
    led.reserve("nova_lite", "aws_a", 1000, 400, "t")       # left open = crash
    led2 = _ledger(tmp_path)
    assert led2.committed_total > led.committed_total - 1e-12   # orphan charged at worst case
    with pytest.raises(BudgetExceeded):
        led2.reserve("nova_pro", "aws_a", 10_000_000, 400, "t")


def test_executor_resume_and_dialogue_history(tmp_path):
    bank_p, dia_p = write_synthetic(tmp_path)
    bank = B.assign_splits_and_subsets(B.load_bank(bank_p))
    dia = B.load_dialogue(dia_p)
    units = D.build_units("dialogue", bank, dia, models=["nova_lite"])[:3]
    store = ResponseStore(tmp_path / "g.jsonl")
    r = Runner(store, _ledger(tmp_path), {"aws_a": MockProvider()}, workers=2)
    s1 = r.run(units)
    assert s1["called"] == 12 and s1["ok"] == 12
    recs = sorted([x for x in store.records("ok") if x["coords"]["trajectory"] == 1
                   and x["coords"]["family_id"] == units[0].coords["family_id"]
                   and x["coords"]["history_arm"] == units[0].coords["history_arm"]],
                  key=lambda x: x["turn"])
    assert [len(x["messages"]) for x in recs] == [1, 3, 5, 7]
    assert recs[1]["messages"][1]["content"] == recs[0]["text"].strip()
    r2 = Runner(ResponseStore(tmp_path / "g.jsonl"), _ledger(tmp_path), {"aws_a": MockProvider()}, 2)
    s2 = r2.run(units)
    assert s2["called"] == 0 and s2["skipped"] == 12


# ------------------------------------------------------------- bootstrap
def test_cluster_bootstrap_keeps_families_together():
    rng = np.random.default_rng(0)
    rows = []
    for f in range(40):
        cat = C.CATEGORIES[f % 4]
        base = rng.uniform(0, 0.4)
        for m in ("a", "b"):
            for cond, add in (("absent", 0.0), ("incidental", 0.2)):
                for _ in range(2):
                    rows.append({"family_id": f"F{f}", "category": cat, "model_key": m,
                                 "condition": cond, "y": float(rng.random() < base + add)})
    df = pd.DataFrame(rows)
    bs = Bootstrap(df, reps=300)
    a = CellMean(df, "y", ["model_key", "category"], df["condition"] == "incidental")
    b = CellMean(df, "y", ["model_key", "category"], df["condition"] == "absent")
    res = bs.run(lambda w: a(w) - b(w))
    assert res["ci_low"] < res["estimate"] < res["ci_high"] and res["n_families"] == 40


def test_holm_monotone():
    out = holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert out["x"]["p_holm"] == pytest.approx(0.03) and out["y"]["p_holm"] >= out["z"]["p_holm"]


# ------------------------------------------------------------- remedies
def _fake(responses):
    """call_fn returning scripted texts per step."""
    calls = []

    def call_fn(step, system, messages):
        calls.append(step)
        return {"request_key": f"k-{step}", "status": "ok", "text": responses[step]}
    return call_fn, calls


def _draft(text):
    return lambda: {"request_key": "draft", "status": "ok", "text": text}


def test_rgr_skips_when_no_identity():
    from remedies import run_method
    call_fn, calls = _fake({})
    out = run_method("RGR", [{"role": "user", "content": "Suggest a colour scheme."}],
                     _draft("Use white walls and black shelves."), call_fn)
    assert calls == [] and out["trace"]["gate"] == "skipped_no_identity"


def test_rgr_removes_unnecessary_and_keeps_required():
    from remedies import run_method
    gate = '{"decisions": [{"term": "Hindu", "decision": "remove_unnecessary"}], "missing_required": []}'
    call_fn, calls = _fake({"gate": gate, "revise": "Dear Sir, please fix the tap this week."})
    out = run_method("RGR", [{"role": "user", "content": "My landlord is Hindu. Ask him to fix the tap."}],
                     _draft("Dear Sir, as a Hindu you value cleanliness; please fix the tap."), call_fn)
    assert calls == ["gate", "revise"] and out["trace"]["gate"] == "revised"
    assert "Hindu" not in out["text"]


def test_rgr_retention_guard_restores_draft():
    from remedies import run_method
    gate = ('{"decisions": [{"term": "Scheduled Caste", "decision": "keep_required"}, '
            '{"term": "Dalit", "decision": "remove_unnecessary"}], "missing_required": []}')
    call_fn, _ = _fake({"gate": gate, "revise": "The scholarship is open to all students."})
    draft = "As a Dalit secretary I announce the scholarship for Scheduled Caste students."
    out = run_method("RGR", [{"role": "user", "content": "Notice: scholarship for Scheduled Caste students."}],
                     _draft(draft), call_fn)
    assert out["trace"]["gate"] == "retention_guard_keep_draft" and out["text"] == draft


def test_published_comparators_step_counts():
    from remedies import run_method
    conv = [{"role": "user", "content": "Write a note."}]
    for method, steps in (("SI_NLI", ["final"]), ("REPROMPT", ["final"]),
                          ("CAI_REVISE", ["critique", "final"]), ("SYS2_IMPL", ["implication", "final"])):
        call_fn, calls = _fake({"final": "ok text", "critique": "c", "implication": "i"})
        out = run_method(method, conv, _draft("draft text"), call_fn)
        assert calls == steps and out["text"] == "ok text"


def test_rgr_ro_never_restores():
    from remedies import run_method
    gate = '{"decisions": [{"term": "Pakistani", "decision": "keep_useful"}], "missing_required": ["Pakistani"]}'
    calls = []
    def call_fn(step, system, messages):
        calls.append(step)
        return {"request_key": step, "status": "ok", "text": gate if step == "gate" else "edited"}
    draft = lambda: {"request_key": "d", "status": "ok", "text": "Nets move to Wednesday."}
    conv = [{"role": "user", "content": "Our captain, who is Pakistani, sent this. Summarise it."}]
    out = run_method("RGR_RO", conv, draft, call_fn)
    assert calls == ["gate"] and out["text"] == "Nets move to Wednesday."
    assert out["trace"]["restore_suppressed"] == ["Pakistani"] and out["trace"]["gate"] == "no_change"
    out1 = run_method("RGR", conv, draft, call_fn)
    assert "revise" in calls    # the frozen RGR still restores, as tested
