#!/usr/bin/env python3
"""
run_all.py - single entry point for the v2 study
"When Identity Belongs in the Answer" (Future_Plan.md, revised 8 Oct 2026).

    python run_all.py                       # status and the next step
    python run_all.py <command> [options]   # see README.md for the full order

Paid commands (smoke, generate, judge-calibrate, judge, judge-third) need
--confirm-spend and run under the ledger ceiling of --stage. --dry-run uses
the mock provider and writes to outputs_dryrun/ at zero cost.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import config_v2 as C  # noqa: E402
import bank as B  # noqa: E402
import design as D  # noqa: E402
import gates as G  # noqa: E402
import judges as J  # noqa: E402
import lexicon_v2 as LX  # noqa: E402
from common import append_jsonl, read_jsonl, setup_logging, utc_now  # noqa: E402
from executor import Runner  # noqa: E402
from ledger import Ledger, load_prices  # noqa: E402
from store import ResponseStore  # noqa: E402

log = logging.getLogger("run_all")


# ================================================================ context
class Ctx:
    def __init__(self, args):
        self.args = args
        self.dry = bool(getattr(args, "dry_run", False))
        self.out = Path(args.out) if getattr(args, "out", None) else (C.DRYRUN_DIR if self.dry else C.OUTPUT_DIR)
        self.out.mkdir(parents=True, exist_ok=True)
        self.bank_path = Path(args.bank) if getattr(args, "bank", None) else C.PROMPT_BANK
        self.dialogue_path = Path(args.dialogue_bank) if getattr(args, "dialogue_bank", None) else C.DIALOGUE_BANK
        self.gates_path = (self.out / "gates.json") if self.dry else C.GATES_FILE
        self.ledger_path = (self.out / "ledger.jsonl") if self.dry else (C.MANIFEST_DIR / "ledger.jsonl")
        self.ann_root = (self.out / "annotation") if self.dry else C.ANNOTATION_DIR
        setup_logging(self.out / "run.log")
        self._point_annotation_dirs()

    def _point_annotation_dirs(self):
        import annotation.make_sheets as MS
        import annotation.reliability as RL
        MS.KEYS_DIR = self.ann_root / "keys"
        MS.DIST_DIR = self.ann_root / "dist"
        MS.REGISTRY = MS.KEYS_DIR / "unit_registry.csv"
        RL.RETURNED_DIR = self.ann_root / "returned"
        RL.REPORT_DIR = self.ann_root / "reports"

    def gen_store(self) -> ResponseStore:
        return ResponseStore(self.out / "generations.jsonl")

    def judge_store(self) -> ResponseStore:
        return ResponseStore(self.out / "judgments.jsonl")

    def bank(self) -> pd.DataFrame:
        return B.load_bank(self.bank_path)

    def dialogue(self) -> pd.DataFrame:
        return B.load_dialogue(self.dialogue_path)

    def ledger(self) -> Ledger:
        stage = self.args.stage
        return Ledger(self.ledger_path, load_prices(C.PRICES_FILE), stage, C.STAGE_CEILINGS,
                      C.HARD_CAP_USD, C.ACCOUNT_CEILINGS)

    def providers(self, accounts: set[str]) -> dict:
        if self.dry:
            from providers.mock import MockProvider
            return {a: MockProvider(a) for a in accounts}
        from providers.bedrock import BedrockProvider
        return {a: BedrockProvider(a) for a in accounts}


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=C.V2_DIR, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _manifest(ctx: Ctx, command: str, extra: dict | None = None) -> None:
    rec = {"ts": utc_now(), "command": command, "dry_run": ctx.dry, "git": _git_commit(),
           "lexicon": LX.LEXICON_VERSION, "judge_prompt": J.JUDGE_PROMPT_VERSION,
           "region": C.REGION, "account_mode": C.ACCOUNT_MODE,
           "models": {k: v["model_id"] for k, v in C.MODELS.items()},
           "args": {k: v for k, v in vars(ctx.args).items() if k != "func"}}
    rec.update(extra or {})
    append_jsonl((ctx.out if ctx.dry else C.MANIFEST_DIR) / "run_manifest.jsonl", rec)


def _need_spend(ctx: Ctx) -> None:
    if not ctx.dry and not ctx.args.confirm_spend:
        sys.exit("This command calls paid Bedrock models. Re-run with --confirm-spend "
                 f"(stage '{ctx.args.stage}', ceiling ${C.STAGE_CEILINGS[ctx.args.stage]:.0f}).")


def _gate(ctx: Ctx, fn) -> None:
    try:
        fn(ctx.gates_path)
    except G.GateError as exc:
        if ctx.dry:
            log.warning("DRY RUN: gate not enforced: %s", exc)
        else:
            sys.exit(f"Gate check failed: {exc}")


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ================================================================ commands
def cmd_status(ctx: Ctx) -> None:
    bank = ctx.bank() if ctx.bank_path.exists() else pd.DataFrame(columns=B.BANK_COLUMNS)
    fam = bank.drop_duplicates("family_id")
    print("== Prompt bank:", ctx.bank_path)
    print(fam.groupby(["split", "category"]).size().unstack(fill_value=0) if len(fam) else "  (empty)")
    print("   review status:", bank["review_status"].value_counts().to_dict())
    dia = ctx.dialogue()
    print("== Dialogue bank:", len(dia), "families; review:", dia["review_status"].value_counts().to_dict())
    gen = ctx.gen_store()
    done = pd.Series([r["experiment"] for r in gen.records("ok")
                      if r.get("role") not in ("method_step",)]).value_counts().to_dict()
    planned = D.design_call_counts()
    planned["comparators"] = D.comparator_call_counts()["finals"]   # count final answers, not calls
    print("== Generation calls done / design (comparators: final answers):")
    for exp, n in planned.items():
        print(f"   {exp:13s} {done.get(exp, 0):6d} / {n}")
    jd = ctx.judge_store()
    print("== Judge calls done:", len(list(jd.records("ok"))))
    print("== Gates:", list(G.load_gates(ctx.gates_path)) or "none")
    print("== Ledger:")
    _print(ctx.ledger().summary())
    print("\nNext step: see README.md 'Run order'. Typical first steps: "
          "'python run_all.py check', then 'python run_all.py generate --experiment pilot "
          "--confirm-spend' after prices and credit coverage are confirmed.")


def cmd_check(ctx: Ctx) -> None:
    bank = ctx.bank()
    mode = ctx.args.mode
    splits = set(ctx.args.splits.split(",")) if ctx.args.splits else None
    rep = B.validate_bank(bank, mode, splits)
    print("== prompt bank", f"({mode})"); print(rep.text())
    drep = B.validate_dialogue(ctx.dialogue(), mode)
    print("== dialogue bank", f"({mode})"); print(drep.text())
    print("== keys present:", {a: bool(_has_keys(a)) for a in C.ACCOUNTS})
    if not (rep.ok and drep.ok):
        sys.exit(1)


def _has_keys(alias: str) -> bool:
    from dotenv import dotenv_values
    env = dotenv_values(C.ENV_PATH) if C.ENV_PATH.exists() else {}
    return all(env.get(v) for v in C.ACCOUNTS[alias])


def cmd_forecast(ctx: Ctx) -> None:
    df = D.forecast(load_prices(C.PRICES_FILE))
    pd.set_option("display.width", 160)
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    gen = df[~df["item"].str.startswith("judge_")]
    print(f"\nGeneration calls {gen['calls'].sum():,}; all calls {df['calls'].sum():,}")
    print(f"Allowance-rate inference total ${df['cost_allowance_usd'].sum():,.2f}; "
          f"tariff-rate ${df['cost_tariff_usd'].sum():,.2f} (Qwen3 32B at allowance).")


def cmd_assign(ctx: Ctx) -> None:
    if "protocol_frozen" in G.load_gates(ctx.gates_path) and not ctx.dry:
        sys.exit("Protocol is frozen; split/subset assignment cannot change.")
    bank = B.assign_splits_and_subsets(ctx.bank(), overwrite=ctx.args.overwrite)
    B.save_bank(bank, ctx.bank_path)
    fam = bank.drop_duplicates("family_id")
    print(fam.groupby(["split", "category"]).size().unstack(fill_value=0))
    print("intervention subset:", int(fam["intervention_subset"].sum()),
          "paraphrase subset:", int(fam["paraphrase_subset"].sum()),
          "schema subset:", int(fam["schema_subset"].sum()))


def cmd_smoke(ctx: Ctx) -> None:
    _need_spend(ctx)
    models = ctx.args.models.split(",") if ctx.args.models else C.GENERATORS
    accounts = {C.PRIMARY_ACCOUNT} if C.ACCOUNT_MODE == "primary" else set(C.ACCOUNTS)
    provs = ctx.providers(accounts)
    store = ResponseStore(ctx.out / "smoke.jsonl")
    runner = Runner(store, ctx.ledger(), provs, workers=2)
    units = []
    for acct in sorted(accounts):
        for m in models:
            units.append(D.Unit("single", "smoke", "smoke", m, acct, "B0",
                                ["Reply with the single word: ready."],
                                {"experiment": "smoke", "date": utc_now()[:10], "account": acct},
                                settings_override={"temperature": 0.0, "max_tokens": 16}))
    stats = runner.run(units)
    for r in store.records(None):
        print(f"{r['account']:6s} {r['model_key']:15s} {r['status']:12s} system_mode={r['system_mode']} "
              f"in={r['input_tokens']} out={r['output_tokens']} text={r['text'][:40]!r} err={r['error']}")
    _manifest(ctx, "smoke", {"stats": stats})


def cmd_generate(ctx: Ctx) -> None:
    _need_spend(ctx)
    exp = ctx.args.experiment
    bank, dia = ctx.bank(), ctx.dialogue()
    if exp == "pilot":
        rep = B.validate_bank(bank, "draft", {"pilot"})
        splits = {"pilot"}
    else:
        splits = {"dev", "locked"} if ctx.args.split == "all" else {ctx.args.split}
        if "locked" in splits or exp == "dialogue":
            _gate(ctx, G.require_protocol_frozen)
        mode = "draft" if ctx.dry else "freeze"
        rep = B.validate_bank(bank, mode, splits) if exp != "dialogue" else B.validate_dialogue(dia, mode)
    if not rep.ok:
        print(rep.text())
        sys.exit("Bank validation failed; fix the bank before generating.")
    if rep.warnings:
        log.warning("Bank warnings (%d); first: %s", len(rep.warnings), rep.warnings[0])
    models = ctx.args.models.split(",") if ctx.args.models else None
    units = D.build_units(exp, bank, dia, None if exp == "pilot" else splits, models)
    if not units:
        sys.exit(f"No units for experiment {exp} with splits {splits}.")
    accounts = {u.account for u in units}
    runner = Runner(ctx.gen_store(), ctx.ledger(), ctx.providers(accounts), ctx.args.workers,
                    ctx.args.retry_ambiguous)
    stats = runner.run(units, ctx.args.limit)
    _print(stats)
    _manifest(ctx, f"generate:{exp}", {"stats": stats})


def cmd_score(ctx: Ctx) -> None:
    from scoring import score_store
    df = score_store(ctx.gen_store(), ctx.out / "scores.jsonl")
    print(f"Scored {len(df)} generations -> {ctx.out / 'scores.csv'}")


def _gen_frame(ctx: Ctx) -> pd.DataFrame:
    from annotation.make_sheets import generation_frame
    return generation_frame(ctx.gen_store())


def cmd_sample(ctx: Ctx) -> None:
    import annotation.make_sheets as MS
    from scoring import load_scores
    gen = _gen_frame(ctx)
    wanted = ctx.args.sample.split(",")
    seed = ctx.args.seed
    reg = MS._registry()
    exclude = set(reg["request_key"])
    parts = []
    for s in wanted:
        if s == "pilot":
            parts.append(MS.sample_pilot(gen, seed))
        elif s == "representative":
            parts.append(MS.sample_representative(gen[~gen["request_key"].isin(exclude)], seed))
        elif s == "challenge":
            taken = exclude | set(pd.concat(parts)["request_key"]) if parts else exclude
            parts.append(MS.sample_challenge(gen, load_scores(ctx.out / "scores.jsonl"), taken, seed))
        elif s == "intervention":
            parts.append(MS.sample_intervention(gen, seed))
        elif s == "rgr2":
            parts.append(MS.sample_intervention(gen, seed, arms=["B0", "B1", "B2", "RGR", "RGR_RO"],
                                                split="dev", sample_name="rgr2_review"))
        else:
            sys.exit(f"unknown sample {s}")
    sample = pd.concat(parts, ignore_index=True)
    if sample.empty:
        sys.exit("Sample is empty: generate the required outputs first.")
    import random
    order = list(sample.index)
    random.Random(seed).shuffle(order)
    sample = sample.loc[order].reset_index(drop=True)
    size = ctx.args.max_per_batch or len(sample)
    batches = []
    for i in range(0, len(sample), size):
        name = ctx.args.batch if len(sample) <= size else f"{ctx.args.batch}_{i // size + 1}"
        out = MS.write_batch(sample.iloc[i:i + size], name, seed)
        batches.append({"batch": name, "units": int(min(size, len(sample) - i)), "dir": str(out)})
    print(sample.groupby("sample").size().to_string())
    _print(batches)
    _manifest(ctx, "sample", {"batches": batches})


def cmd_reliability(ctx: Ctx) -> None:
    import annotation.reliability as RL
    res = RL.run(ctx.args.batch)
    _print(res)
    print((Path(res["report_dir"]) / "reliability.md").read_text(encoding="utf-8"))


def load_human_labels(ctx: Ctx) -> pd.DataFrame:
    """All batches: key file + majority labels (+ adjudication overrides)."""
    import annotation.reliability as RL
    keys_dir = ctx.ann_root / "keys"
    frames = []
    for key_file in sorted(keys_dir.glob("*_key.csv")):
        batch = key_file.name[:-len("_key.csv")]
        maj_file = RL.REPORT_DIR / batch / "majority_labels.csv"
        if not maj_file.exists():
            continue
        key = pd.read_csv(key_file, dtype=str)
        maj = pd.read_csv(maj_file, dtype=str)
        adj_file = RL.REPORT_DIR / batch / "adjudicated.csv"
        maj["maj_unnecessary_raw"] = maj["maj_unnecessary"]
        maj["maj_required_identity_retained_raw"] = maj["maj_required_identity_retained"]
        if adj_file.exists():
            adj = pd.read_csv(adj_file, dtype=str).set_index("unit_id")
            maj["maj_unnecessary"] = maj["unit_id"].map(adj["adjudicated_unnecessary"]).fillna(maj["maj_unnecessary"])
            # Adjudication applies to every outcome, not only the primary label.
            for f in ("unsupported_personal_attribution", "task_adequacy", "required_identity_retained", "refusal"):
                if f in adj.columns:
                    maj[f"maj_{f}"] = maj["unit_id"].map(adj[f]).fillna(maj[f"maj_{f}"])
        frames.append(key.merge(maj, on="unit_id").assign(batch=batch))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cmd_judge_calibrate(ctx: Ctx) -> None:
    _need_spend(ctx)
    keys_dir = ctx.ann_root / "keys"
    keys = pd.concat([pd.read_csv(f, dtype=str) for f in keys_dir.glob("*_key.csv")]) \
        if any(keys_dir.glob("*_key.csv")) else pd.DataFrame()
    if keys.empty:
        sys.exit("No annotation key files: create annotation batches first.")
    gen = ctx.gen_store()
    targets = [gen.get(k) for k in keys["request_key"].unique() if gen.get(k)]
    units = J.calibration_units(targets)
    runner = Runner(ctx.judge_store(), ctx.ledger(), ctx.providers({u.account for u in units}),
                    ctx.args.workers, ctx.args.retry_ambiguous)
    stats = runner.run(units, ctx.args.limit)
    _print(stats)
    _manifest(ctx, "judge-calibrate", {"stats": stats, "targets": len(targets)})


def cmd_audit(ctx: Ctx) -> None:
    from analysis.evaluator_audit import run as audit
    from scoring import load_scores
    human = load_human_labels(ctx)
    if human.empty:
        sys.exit("No human majority labels found (run 'reliability' for returned batches).")
    gen = _gen_frame(ctx)[["request_key", "model_key"]]
    scores = load_scores(ctx.out / "scores.jsonl")
    units = human.merge(gen, on="request_key", how="left", suffixes=("", "_g"))
    units["model_key"] = units["model_key"].fillna(units.get("model_key_g"))
    units = units.merge(scores, on="request_key", how="left")
    units["human"] = units["maj_unnecessary"]
    jdf = J.judgments_frame(list(ctx.judge_store().records("ok")))
    calib = jdf[jdf["purpose"] == "calibration"]
    path = (ctx.out / "evaluator_audit.json") if ctx.dry else (C.MANIFEST_DIR / "evaluator_audit.json")
    rep = audit(units, calib, path)
    _print({"gate_check": rep.get("gate_check"), "best_baseline_on_dev": rep.get("best_baseline_on_dev"),
            "added_value": rep.get("added_value"), "report": str(path)})


def cmd_gate(ctx: Ctx) -> None:
    if ctx.args.show or not ctx.args.set:
        _print(G.load_gates(ctx.gates_path))
        return
    name = ctx.args.set
    evidence = {}
    if name == "protocol_frozen":
        rep = B.validate_bank(ctx.bank(), "freeze", {"dev", "locked"})
        drep = B.validate_dialogue(ctx.dialogue(), "freeze")
        if not (rep.ok and drep.ok) and not ctx.args.override:
            print(rep.text()); print(drep.text())
            sys.exit("Banks do not pass freeze validation; fix them or use --override with a note.")
    elif name == "evaluator_validated":
        path = (ctx.out / "evaluator_audit.json") if ctx.dry else (C.MANIFEST_DIR / "evaluator_audit.json")
        if not path.exists():
            sys.exit("Run 'audit' first.")
        evidence = json.loads(path.read_text(encoding="utf-8")).get("gate_check", {})
        ok = evidence.get("meets_point_targets") and evidence.get("enough_cases")
        if not ok and not ctx.args.override:
            _print(evidence)
            sys.exit("Evaluator targets not met. Narrow the claims (Plan s12) or use --override "
                     "with a note explaining the decision.")
    if ctx.args.override:
        evidence["override"] = True
    entry = G.set_gate(name, ctx.args.note or "", evidence, ctx.gates_path)
    _print({name: entry})


def cmd_judge(ctx: Ctx) -> None:
    _need_spend(ctx)
    _gate(ctx, G.require_evaluator_validated)
    targets = [r for r in ctx.gen_store().records("ok") if r["experiment"] in J.JUDGED_EXPERIMENTS]
    units = J.production_units(targets)
    runner = Runner(ctx.judge_store(), ctx.ledger(), ctx.providers({u.account for u in units}),
                    ctx.args.workers, ctx.args.retry_ambiguous)
    stats = runner.run(units, ctx.args.limit)
    _print(stats)
    _manifest(ctx, "judge", {"stats": stats, "targets": len(targets)})


def cmd_judge_secondary(ctx: Ctx) -> None:
    """Automated labels for SECONDARY, error-qualified comparisons only (evaluator gate not met).
    Targets: every arm on the intervention-subset families (dev and locked), excluding the pilot."""
    _need_spend(ctx)
    bank = ctx.bank()
    sub = set(bank[bank["intervention_subset"]]["family_id"])
    targets = []
    for r in ctx.gen_store().records("ok"):
        c = r.get("coords") or {}
        if r.get("role") not in ("generation", "final") or c.get("family_id") not in sub:
            continue
        if (r["experiment"] == "single" and c.get("arm") == "B0") or r["experiment"] in ("intervention", "comparators"):
            targets.append(r)
    units = J.production_units(targets, purpose="secondary")
    print(f"secondary judging: {len(targets)} outputs, {len(units)} judge calls")
    runner = Runner(ctx.judge_store(), ctx.ledger(), ctx.providers({u.account for u in units}),
                    ctx.args.workers, ctx.args.retry_ambiguous)
    stats = runner.run(units, ctx.args.limit)
    _print(stats)
    _manifest(ctx, "judge-secondary", {"stats": stats, "targets": len(targets)})


def cmd_judge_third(ctx: Ctx) -> None:
    _need_spend(ctx)
    if not getattr(ctx.args, "secondary", False):
        _gate(ctx, G.require_evaluator_validated)
    gen_by_key = {r["request_key"]: r for r in ctx.gen_store().records("ok")}
    jstore = ctx.judge_store()
    jdf = J.judgments_frame(list(jstore.records("ok")))
    n_judged = sum(1 for r in gen_by_key.values() if r["experiment"] in J.JUDGED_EXPERIMENTS)
    already = int((jdf["purpose"] == "third").sum()) if len(jdf) else 0
    units, info = J.third_vote_units(gen_by_key, jdf, n_judged, already)
    _print(info)
    if units:
        runner = Runner(jstore, ctx.ledger(), ctx.providers({u.account for u in units}),
                        ctx.args.workers, ctx.args.retry_ambiguous)
        _print(runner.run(units, ctx.args.limit))
    _manifest(ctx, "judge-third", {"info": info})


def cmd_consensus(ctx: Ctx) -> None:
    jdf = J.judgments_frame(list(ctx.judge_store().records("ok")))
    jdf.to_csv(ctx.out / "judgments.csv", index=False, encoding="utf-8-sig")
    cons = J.consensus(jdf)
    cons.to_csv(ctx.out / "judge_consensus.csv", index=False, encoding="utf-8-sig")
    print(f"{len(jdf)} judgments; {len(cons)} targets; primary label counts:")
    print(cons["judge_unnecessary"].value_counts().to_string() if len(cons) else "(none)")


def cmd_analyze(ctx: Ctx) -> None:
    from analysis.effects import analyse, build_frame, primary_holm, to_table
    gen = _gen_frame(ctx)
    if ctx.args.labels == "judge":
        path = ctx.out / "judge_consensus.csv"
        if not path.exists():
            sys.exit("Run 'consensus' first.")
        labels = pd.read_csv(path, dtype=str)
    else:
        labels = load_human_labels(ctx)
        if labels.empty:
            sys.exit("No human labels.")
        labels = labels[~labels["sample"].isin(["challenge", "pilot"])]   # never for prevalence
    df = build_frame(gen, labels, ctx.args.labels, ctx.bank())
    res = analyse(df, ctx.args.split)
    audit_path = (ctx.out / "evaluator_audit.json") if ctx.dry else (C.MANIFEST_DIR / "evaluator_audit.json")
    res["holm_primary"] = primary_holm(res, audit_path)
    out = ctx.out / "analysis" / f"{ctx.args.labels}_{ctx.args.split}"
    out.mkdir(parents=True, exist_ok=True)
    to_table(res).to_csv(out / "estimates.csv", index=False, encoding="utf-8-sig")
    with open(out / "results.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(to_table(res).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    _print({"holm_primary": res["holm_primary"], "noninferiority": res.get("noninferiority_vs_B0")})


def cmd_robustness(ctx: Ctx) -> None:
    from analysis.robustness import run as robustness
    labels = load_human_labels(ctx)
    labels = labels[~labels["sample"].isin(["challenge", "pilot"])]
    res = robustness(_gen_frame(ctx), labels, ctx.bank(), ctx.ann_root, ctx.out / "generations.jsonl",
                     ctx.out / "analysis" / "robustness")
    _print({k: (v if k != "rgr_mechanism" else "...") for k, v in res.items()})


def cmd_power(ctx: Ctx) -> None:
    from analysis.power import estimate_sigma, grid
    human = load_human_labels(ctx)
    pilot = human[human["sample"] == "pilot"] if len(human) else human
    sigma_floor = 0.5          # 60 pilot units cannot estimate between-family spread reliably
    if pilot.empty:
        bases = {"assumed": (ctx.args.p0, ctx.args.sigma)}
        print(f"No pilot labels; using --p0 {ctx.args.p0} --sigma {ctx.args.sigma}")
    else:
        pilot = pilot[pilot["maj_unnecessary"].isin(["yes", "no"])].assign(
            y=lambda d: (d["maj_unnecessary"] == "yes").astype(float))
        bases = {}
        for name, sub in (("non_required", pilot[pilot["condition"] != "required"]),
                          ("incidental", pilot[pilot["condition"] == "incidental"])):
            p0 = float(sub["y"].mean())
            sig = max(estimate_sigma(sub), sigma_floor)
            bases[name] = (p0, sig)
            print(f"Pilot {name}: p0={p0:.3f} from {len(sub)} units; sigma(logit) used={sig:.2f}")
    for name, (p0, sig) in bases.items():
        if p0 <= 0:
            print(f"{name}: no positives; power cannot be simulated")
            continue
        print()
        print(f"== {name} (absolute reductions above {100 * p0:.1f} points are impossible)")
        print(grid(p0, sig).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def cmd_method_cost(ctx: Ctx) -> None:
    from analysis.method_cost import method_cost
    df = method_cost(ctx.gen_store())
    if df.empty:
        sys.exit("No remedy outputs yet.")
    df.to_csv(ctx.out / "method_cost.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("RGR gate outcomes:", df.attrs.get("rgr_gate_outcomes"))


def cmd_tables(ctx: Ctx) -> None:
    from analysis.latex_tables import TABLE_DIR, write_all
    if ctx.dry:
        sys.exit("Tables are written only from paid-run outputs (never from mock data).")
    names = write_all(ctx.out, ctx.ann_root, C.MANIFEST_DIR)
    print(f"Wrote {len(names)} tables to {TABLE_DIR}: {names}")


def cmd_cost(ctx: Ctx) -> None:
    _print(ctx.ledger().summary())


def cmd_export(ctx: Ctx) -> None:
    n1 = ctx.gen_store().export_csv(ctx.out / "generations.csv")
    n2 = ctx.judge_store().export_csv(ctx.out / "judgments_raw.csv")
    print(f"generations.csv {n1} rows; judgments_raw.csv {n2} rows in {ctx.out}")


def cmd_salvage(ctx: Ctx) -> None:
    from analysis.salvage_may2026 import run
    print(run(ctx.out / "salvage_may2026").to_string(index=False))


def cmd_pack(ctx: Ctx) -> None:
    from annotation.build_pack import build
    _print(build(Path(ctx.args.dest) if ctx.args.dest else None))


# ================================================================ CLI
def main(argv=None) -> None:
    def add_common(parser, suppress: bool) -> None:
        d = (lambda v: argparse.SUPPRESS) if suppress else (lambda v: v)
        parser.add_argument("--dry-run", action="store_true", default=d(False),
                            help="mock provider, outputs_dryrun/, no spend")
        parser.add_argument("--stage", choices=list(C.STAGE_CEILINGS), default=d("access_pilot"))
        parser.add_argument("--workers", type=int, default=d(C.DEFAULT_WORKERS))
        parser.add_argument("--bank", default=d(None), help="prompt bank CSV")
        parser.add_argument("--dialogue-bank", default=d(None), help="dialogue bank CSV")
        parser.add_argument("--out", default=d(None), help="output directory override")
        parser.add_argument("--confirm-spend", action="store_true", default=d(False))
        parser.add_argument("--retry-ambiguous", action="store_true", default=d(False))
        parser.add_argument("--limit", type=int, default=d(None), help="run at most N units")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(p, suppress=False)
    sub = p.add_subparsers(dest="command")
    _orig_add = sub.add_parser

    def _add(name, **kw):
        sp = _orig_add(name, **kw)
        add_common(sp, suppress=True)
        return sp
    sub.add_parser = _add
    sub.add_parser("status").set_defaults(func=cmd_status)
    s = sub.add_parser("check"); s.add_argument("--mode", default="draft", choices=["draft", "freeze"])
    s.add_argument("--splits", help="comma list, e.g. dev,locked"); s.set_defaults(func=cmd_check)
    sub.add_parser("forecast").set_defaults(func=cmd_forecast)
    s = sub.add_parser("assign"); s.add_argument("--overwrite", action="store_true"); s.set_defaults(func=cmd_assign)
    s = sub.add_parser("smoke"); s.add_argument("--models"); s.set_defaults(func=cmd_smoke)
    s = sub.add_parser("generate")
    s.add_argument("--experiment", required=True, choices=D.EXPERIMENTS)
    s.add_argument("--split", default="dev", choices=["dev", "locked", "all"])
    s.add_argument("--models"); s.set_defaults(func=cmd_generate)
    sub.add_parser("score").set_defaults(func=cmd_score)
    s = sub.add_parser("sample")
    s.add_argument("--sample", required=True, help="pilot | representative,challenge,intervention")
    s.add_argument("--batch", required=True); s.add_argument("--seed", type=int, default=C.SUBSET_SEED)
    s.add_argument("--max-per-batch", type=int); s.set_defaults(func=cmd_sample)
    s = sub.add_parser("reliability"); s.add_argument("--batch", required=True); s.set_defaults(func=cmd_reliability)
    sub.add_parser("judge-calibrate").set_defaults(func=cmd_judge_calibrate)
    sub.add_parser("audit").set_defaults(func=cmd_audit)
    s = sub.add_parser("gate"); s.add_argument("--set", choices=["protocol_frozen", "evaluator_validated"])
    s.add_argument("--note"); s.add_argument("--override", action="store_true")
    s.add_argument("--show", action="store_true"); s.set_defaults(func=cmd_gate)
    sub.add_parser("judge").set_defaults(func=cmd_judge)
    s = sub.add_parser("judge-third"); s.add_argument("--secondary", action="store_true")
    s.set_defaults(func=cmd_judge_third)
    sub.add_parser("judge-secondary").set_defaults(func=cmd_judge_secondary)
    sub.add_parser("consensus").set_defaults(func=cmd_consensus)
    s = sub.add_parser("analyze"); s.add_argument("--labels", default="judge", choices=["judge", "human"])
    s.add_argument("--split", default="locked", choices=["locked", "dev", "all"]); s.set_defaults(func=cmd_analyze)
    s = sub.add_parser("power"); s.add_argument("--p0", type=float, default=0.10)
    s.add_argument("--sigma", type=float, default=0.8); s.set_defaults(func=cmd_power)
    sub.add_parser("cost").set_defaults(func=cmd_cost)
    sub.add_parser("method-cost").set_defaults(func=cmd_method_cost)
    sub.add_parser("robustness").set_defaults(func=cmd_robustness)
    sub.add_parser("tables").set_defaults(func=cmd_tables)
    sub.add_parser("export").set_defaults(func=cmd_export)
    sub.add_parser("salvage").set_defaults(func=cmd_salvage)
    s = sub.add_parser("pack"); s.add_argument("--dest"); s.set_defaults(func=cmd_pack)
    args = p.parse_args(argv)
    if not args.command:
        args.command, args.func = "status", cmd_status
    ctx = Ctx(args)
    args.func(ctx)


if __name__ == "__main__":
    main()
