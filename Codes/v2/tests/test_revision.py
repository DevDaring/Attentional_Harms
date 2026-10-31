"""Tests for the revision analyses (review s6: pairing, exclusions, weighting, calculations).
They read the real ledger and results; skipped when those files are absent (e.g. dry-run checkouts)."""
import json
import math
from pathlib import Path

import pandas as pd
import pytest

REV = Path(__file__).resolve().parent.parent / "revision"
LEDGER = REV / "unit_ledger.csv"
RES = REV / "results" / "reanalysis.json"
need = pytest.mark.skipif(not (LEDGER.exists() and RES.exists()), reason="revision outputs not built")


def _rv():
    import importlib
    import sys
    sys.path.insert(0, str(REV.parent))
    return importlib.import_module("revision.reanalysis")


def test_zero_loss_bound_formula():
    rv = _rv()
    for n in (9, 10, 12, 59):
        assert abs(rv.cp_upper_one_sided(0, n) - (1 - 0.05 ** (1 / n))) < 1e-9
    assert math.ceil(math.log(0.05) / math.log(0.95)) == 59


def test_newcombe_not_degenerate_with_no_discordance():
    rv = _rv()
    d, lo, hi = rv.newcombe_paired(2, 0, 0, 27)          # B1 vs B0: 2 concordant positives, 27 negatives
    assert d == 0 and lo < 0 < hi and abs(lo + hi) < 1e-9


@need
def test_ledger_units_unique_and_fully_rated():
    L = pd.read_csv(LEDGER, dtype=str)
    assert len(L) == 1200 and L.request_key.is_unique and L.unit_id.is_unique
    for r in ("R1", "R2", "R3"):
        assert L[f"{r}_unnecessary"].notna().all()


@need
def test_remedy_pairs_share_b0_coordinates():
    L = pd.read_csv(LEDGER, dtype=str)
    m = L[L["sample"].isin(["intervention", "comparator_review"]) & (L.split == "locked")].copy()
    m["d"] = m.draw.astype(float).astype(int)
    keys = ["family_id", "condition", "model_key", "d"]
    b0 = set(map(tuple, m[m.arm == "B0"][keys].values))
    assert len(b0) == 44
    for arm, g in m[m.arm != "B0"].groupby("arm"):
        assert set(map(tuple, g[keys].values)) == b0, arm


@need
def test_prevalence_uses_representative_sample_only_and_reproduces():
    R = json.loads(RES.read_text(encoding="utf-8"))["prevalence"]
    assert R["n_units"] == 240
    inc = R["by_condition"]["incidental"]
    assert inc["positives"] == 11 and inc["units"] == 80
    assert abs(inc["estimate_weighted_cellmean"] - 0.13194444) < 1e-6        # manuscript 13.2%
    assert abs(R["incidental_minus_absent"]["estimate_weighted"] - 0.12152778) < 1e-6


@need
def test_retention_losses_match_manuscript():
    R = {r["arm"]: r for r in json.loads(RES.read_text(encoding="utf-8"))["remedies_retention"]}
    assert (R["SYS2_IMPL"]["losses"], R["SYS2_IMPL"]["eligible_pairs_original_rule"]) == (6, 9)
    assert (R["CAI_REVISE"]["losses"], R["CAI_REVISE"]["eligible_pairs_original_rule"]) == (3, 10)
    assert (R["IF_MSC"]["losses"], R["IF_MSC"]["eligible_pairs_original_rule"]) == (1, 10)
