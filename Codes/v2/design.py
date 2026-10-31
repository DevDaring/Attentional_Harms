"""
design.py - turns the prompt bank into request units for each experiment and
forecasts calls and cost (Future_Plan.md s5, s9.2, s9.3).

Experiments
  pilot         pilot families x 3 conditions x 2 draws x PILOT_MODELS, arm B0
  single        dev/locked families x 3 conditions x 2 draws x 6 models, arm B0
  intervention  intervention-subset families x 3 conditions x {B1,B2} x 2 draws x 6 models
                (B0 outputs are reused from 'single')
  paraphrase    paraphrase-subset families x 3 conditions x 1 paraphrase x 2 draws x 6 models
  schema        schema-subset families x 3 conditions x old JSON schema x 2 draws x 2 models
  dialogue      dialogue families x {identity, neutral} history x 2 trajectories
                x (opening + 3 switched tasks) x 6 models, real replies fed back
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import config_v2 as C
from bank import account_for_family

EXPERIMENTS = ["pilot", "single", "intervention", "paraphrase", "schema", "dialogue", "comparators",
               "rgr_ro", "followup_matched"]


@dataclass
class Unit:
    """One request (kind='single') or one dialogue trajectory (kind='trajectory')."""
    kind: str
    phase: str
    experiment: str
    model_key: str
    account: str
    system_id: str
    user_turns: list[str]               # 1 for single; opening + switched tasks for trajectory
    coords: dict
    meta: dict = field(default_factory=dict)
    role: str = "generation"            # or "judge"
    system_override: str | None = None  # judge prompts live in judges.py
    settings_override: dict | None = None

    @property
    def system(self) -> str:
        return self.system_override if self.system_override is not None else C.SYSTEM_PROMPTS[self.system_id]

    @property
    def settings(self) -> dict:
        return self.settings_override or C.GEN_SETTINGS

    @property
    def n_calls(self) -> int:
        if self.kind == "method":
            from remedies import EXTRA_CALLS_MAX
            return EXTRA_CALLS_MAX[self.system_id]      # worst case; the B0 draft is reused
        return len(self.user_turns)


def _family_accounts(df: pd.DataFrame) -> dict[str, str]:
    fam = df.drop_duplicates("family_id")
    return account_for_family(list(fam["family_id"]), dict(zip(fam["family_id"], fam["category"])))


def _single_units(rows: pd.DataFrame, experiment: str, models: list[str], arms: list[str],
                  text_col: str = "prompt", variant: str = "original") -> list[Unit]:
    accounts = _family_accounts(rows)
    units = []
    for _, r in rows.sort_values(["family_id", "condition"]).iterrows():
        for model in models:
            for arm in arms:
                for draw in range(1, C.N_DRAWS + 1):
                    coords = {"experiment": experiment, "family_id": r["family_id"],
                              "condition": r["condition"], "arm": arm, "draw": draw,
                              "prompt_variant": variant}
                    meta = {"category": r["category"], "split": r["split"],
                            "task_type": r["task_type"], "identity_detail": r["identity_detail"]}
                    units.append(Unit("single", experiment, experiment, model,
                                      accounts[r["family_id"]], arm, [r[text_col]], coords, meta))
    return units


def build_units(experiment: str, bank: pd.DataFrame, dialogue: pd.DataFrame | None = None,
                splits: set[str] | None = None, models: list[str] | None = None) -> list[Unit]:
    if experiment == "pilot":
        rows = bank[bank["split"] == "pilot"]
        return _single_units(rows, "pilot", models or C.PILOT_MODELS, ["B0"])
    core = bank[bank["split"].isin(splits or {"dev", "locked"})]
    if experiment == "single":
        return _single_units(core, "single", models or C.GENERATORS, ["B0"])
    if experiment == "intervention":
        return _single_units(core[core["intervention_subset"]], "intervention",
                             models or C.GENERATORS, ["B1", "B2"])
    if experiment == "paraphrase":
        return _single_units(core[core["paraphrase_subset"]], "paraphrase",
                             models or C.GENERATORS, ["B0"], text_col="paraphrase",
                             variant="paraphrase_1")
    if experiment == "schema":
        return _single_units(core[core["schema_subset"]], "schema",
                             models or C.SCHEMA_ABLATION_MODELS, ["SCHEMA_OLD"])
    if experiment == "comparators":
        from remedies import COMPARATORS, PROPOSED
        sub = bank[bank["intervention_subset"]]
        units = []
        for method in COMPARATORS + PROPOSED:
            allowed = C.RGR_SPLITS if method in PROPOSED else C.COMPARATOR_SPLITS
            rows = sub[sub["split"].isin(allowed & (splits or {"dev", "locked"}))]
            for u in _single_units(rows, "comparators", models or C.GENERATORS, [method]):
                u.kind = "method"
                u.system_id = method
                units.append(u)
        return units
    if experiment == "followup_matched":
        # Follow-up exploratory arms (remedies.FOLLOWUP) on exactly the 44 human-labelled locked
        # matched sets (same family, condition, model and draw as the labelled B0 outputs).
        from remedies import FOLLOWUP
        keys = pd.concat([pd.read_csv(C.ANNOTATION_DIR / "keys" / f"main_{i}_key.csv", dtype=str)
                          for i in (1, 2, 3)])
        sets = keys[(keys["sample"] == "intervention") & (keys["arm"] == "B0")]
        want = {(r.family_id, r.condition, r.model_key, int(float(r.draw))) for r in sets.itertuples()}
        arms = [a for a in FOLLOWUP if a != "CI_REASON" or __import__("remedies").CI_REASON_SYSTEM]
        units = []
        rows = bank[bank["family_id"].isin({w[0] for w in want}) & (bank["split"] == "locked")]
        for method in arms:
            for u in _single_units(rows, "comparators", models or C.GENERATORS, [method]):
                c = u.coords
                if (c["family_id"], c["condition"], u.model_key, int(c["draw"])) in want:
                    u.kind, u.system_id = "method", method
                    units.append(u)
        return units
    if experiment == "rgr_ro":
        # Post-hoc remove-only RGR: development intervention families only (never locked).
        rows = bank[bank["intervention_subset"] & (bank["split"] == "dev")]
        units = _single_units(rows, "comparators", models or C.GENERATORS, ["RGR_RO"])
        for u in units:
            u.kind, u.system_id = "method", "RGR_RO"
        return units
    if experiment == "dialogue":
        if dialogue is None or dialogue.empty:
            return []
        d = dialogue if not splits else dialogue[dialogue["split"].isin(splits) |
                                                 (dialogue["split"] == "")]
        accounts = account_for_family(list(d["family_id"]), dict(zip(d["family_id"], d["category"])))
        units = []
        for _, r in d.sort_values("family_id").iterrows():
            switches = [r["switch_task_1"], r["switch_task_2"], r["switch_task_3"]]
            for model in models or C.GENERATORS:
                for history_arm, opening in (("identity", r["identity_opening"]),
                                             ("neutral", r["neutral_opening"])):
                    for traj in range(1, C.DIALOGUE_TRAJECTORIES + 1):
                        coords = {"experiment": "dialogue", "family_id": r["family_id"],
                                  "history_arm": history_arm, "trajectory": traj, "arm": "B0"}
                        meta = {"category": r["category"], "split": r["split"] or "locked",
                                "identity_detail": r["identity_detail"]}
                        units.append(Unit("trajectory", "dialogue", "dialogue", model,
                                          accounts[r["family_id"]], "B0",
                                          [opening] + switches, coords, meta))
        return units
    raise ValueError(f"unknown experiment {experiment!r}")


# ---------------------------------------------------------------- forecasts
def design_call_counts() -> dict[str, int]:
    """Calls implied by the design constants alone (Plan s9.2 table)."""
    D, nd, nm = C.DESIGN, C.N_DRAWS, len(C.GENERATORS)
    nc = len(C.CONDITIONS)
    core = D["dev_families"] + D["locked_families"]
    inter = D["intervention_subset"]["dev"] + D["intervention_subset"]["locked"]
    return {
        "pilot": D["pilot_families"] * nc * nd * len(C.PILOT_MODELS),
        "single": core * nc * nd * nm,
        "dialogue": D["dialogue_families"] * 2 * C.DIALOGUE_TRAJECTORIES * (1 + C.DIALOGUE_SWITCH_TURNS) * nm,
        "intervention": inter * nc * 2 * nd * nm,
        "paraphrase": D["paraphrase_subset"] * nc * 1 * nd * nm,
        "schema": D["schema_subset"] * nc * nd * len(C.SCHEMA_ABLATION_MODELS),
        "comparators": comparator_call_counts()["total_calls"],
    }


def comparator_call_counts() -> dict:
    """Worst-case extra calls of the remedy methods (B0 drafts are reused)."""
    from remedies import COMPARATORS, EXTRA_CALLS_MAX, PROPOSED
    D, nd, nm, nc = C.DESIGN, C.N_DRAWS, len(C.GENERATORS), len(C.CONDITIONS)
    fam = {"dev": D["intervention_subset"]["dev"], "locked": D["intervention_subset"]["locked"]}
    single = multi = finals = 0
    for m in COMPARATORS + PROPOSED:
        splits = C.RGR_SPLITS if m in PROPOSED else C.COMPARATOR_SPLITS
        n_final = sum(fam[s] for s in splits) * nc * nd * nm
        finals += n_final
        if m in ("SI_NLI", "IF_MSC", "DAIQ_GUARD"):
            single += n_final
        else:
            multi += n_final * EXTRA_CALLS_MAX[m]
    return {"single_calls": single, "multi_calls": multi, "total_calls": single + multi, "finals": finals}


def judge_call_counts(gen_counts: dict[str, int]) -> dict[str, int]:
    judged = sum(gen_counts[e] for e in ("single", "dialogue", "intervention", "paraphrase"))
    if "comparators" in gen_counts:
        judged += comparator_call_counts()["finals"]
    return {"judged_outputs": judged, "two_judges": 2 * judged,
            "third_vote_reserve": int(round(C.THIRD_JUDGE_RESERVE_FRACTION * judged)),
            "calibration": (C.ANNOTATION["pilot_units"] + C.ANNOTATION["representative_single"]
                            + C.ANNOTATION["representative_dialogue"] + C.ANNOTATION["challenge_units"]
                            + len(C.INTERVENTION_REVIEW_ARMS) * C.ANNOTATION["intervention_triplets"])
            * len(C.JUDGE_CALIBRATION_CANDIDATES)}


def forecast(prices: dict | None = None) -> pd.DataFrame:
    """Planning cost at the allowance rate (and at listed tariffs when given)."""
    from ledger import cost, rates_for
    gen = design_call_counts()
    allow = C.ALLOWANCE_RATE
    rows = []
    for exp, n in gen.items():
        if exp == "comparators":
            cc = comparator_call_counts()
            for item, n2, key in (("comparators_single", cc["single_calls"], "comparators_single"),
                                  ("comparators_multi", cc["multi_calls"], "comparators_multi")):
                tin, tout = C.TOKEN_ASSUMPTIONS[key]
                rows.append({"item": item, "calls": n2, "input_tokens": n2 * tin,
                             "output_tokens": n2 * tout,
                             "cost_allowance_usd": cost(allow, n2 * tin, n2 * tout)})
            continue
        tin, tout = C.TOKEN_ASSUMPTIONS[exp]
        rows.append({"item": exp, "calls": n, "input_tokens": n * tin, "output_tokens": n * tout,
                     "cost_allowance_usd": cost(allow, n * tin, n * tout)})
    j = judge_call_counts(gen)
    tin, tout = C.TOKEN_ASSUMPTIONS["judge"]
    for item in ("two_judges", "third_vote_reserve", "calibration"):
        n = j[item]
        rows.append({"item": f"judge_{item}", "calls": n, "input_tokens": n * tin,
                     "output_tokens": n * tout, "cost_allowance_usd": cost(allow, n * tin, n * tout)})
    df = pd.DataFrame(rows)
    if prices is not None:
        # Tariff view: generation spread evenly across the models of each experiment.
        tariff_costs = []
        for _, r in df.iterrows():
            item = r["item"]
            if item == "pilot":
                models = C.PILOT_MODELS
            elif item == "schema":
                models = C.SCHEMA_ABLATION_MODELS
            elif item.startswith("judge_"):
                models = C.JUDGE_CALIBRATION_CANDIDATES if item == "judge_calibration" else \
                    C.JUDGE_PRIMARY + [C.JUDGE_SUBSTITUTE]
            else:
                models = C.GENERATORS
            share = 1 / len(models)
            tariff_costs.append(sum(cost(rates_for(prices, m)[1], r["input_tokens"] * share,
                                         r["output_tokens"] * share) for m in models))
        df["cost_tariff_usd"] = tariff_costs
    return df
