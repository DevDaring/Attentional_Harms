# v2 pipeline: When Identity Belongs in the Answer

Code for the redesigned study in `Future_Plan.md` (revised 8 October 2026). Every request goes
through one entry point, `run_all.py`. Runs resume from `outputs/`, every paid call reserves its
worst-case cost in a ledger first, and locked-split work is blocked until the protocol is frozen.

```
cd Codes/v2
pip install -r requirements_v2.txt
python run_all.py            # status and next step
python -m pytest tests -q    # 24 unit tests
python tests/e2e_dry_run.py  # full rehearsal on a synthetic bank, mock models, zero cost
```

## Run order

| Step | Command | Paid | Notes |
|---|---|---|---|
| 1 | `check` / `forecast` | no | Bank validation and call/cost forecast |
| 2 | `smoke --confirm-spend` | ~US$0.001 | One tiny call per model on the primary account. Done 8 Oct |
| 3 | `generate --experiment pilot --confirm-spend` | ~US$0.05 | 288 calls. Done 8 Oct |
| 4 | `score` | no | Lexical baselines for every output |
| 5 | `sample --sample pilot --batch pilot`, then `pack` | no | Annotator batch and zips. Done 8 Oct |
| 6 | `reliability --batch pilot` | no | After the pilot sheets come back into `annotation/returned/pilot/` |
| 7 | `power` | no | Planning simulation from pilot labels |
| 8 | review banks, `check --mode freeze` | no | 240 core + 80 dialogue families drafted and assigned on 8 Oct; two human reviews each still needed |
| 9 | `generate --experiment single --split dev --stage validation --confirm-spend` | yes | Dev runs before freezing |
| 10 | `gate --set protocol_frozen --note "..."` | no | Hashes protocol, banks, config, prices |
| 11 | `generate --experiment single|intervention|paraphrase|schema|dialogue|comparators --split locked --stage core --confirm-spend` | yes | Locked runs; `comparators` runs the six published remedies and RGR (`remedies.py`) |
| 12 | `sample --sample representative,challenge,intervention --batch main --max-per-batch 220` | no | Main annotation batches |
| 13 | `judge-calibrate --stage core --confirm-spend` | yes | All judge candidates on human-labelled units |
| 14 | `audit`, then `gate --set evaluator_validated` | no | Refuses unless the evaluator targets are met, or an override is recorded |
| 15 | `judge`, `judge-third`, `consensus` | yes | Production judging with the 10% third-vote cap |
| 16 | `analyze --labels judge` and `analyze --labels human` | no | Family-cluster bootstrap, Holm on the primary family |
| 17 | `method-cost`, `tables` | no | Remedy cost per answer; writes the manuscript tables into `Submission/ipm_tables/` |
| 18 | `cost`, `export`, `salvage` | no | Ledger summary, CSV export, May 2026 reconstruction |

Global options work before or after the command: `--dry-run`, `--stage
{access_pilot,validation,core,reruns}`, `--workers`, `--limit`,
`--confirm-spend`, `--retry-ambiguous`. Stage ceilings are US$10/60/370/450 since the comparators were added.

## Layout

| Path | Purpose |
|---|---|
| `config_v2.py` | Every design constant, model ID, prompt, ceiling and gate target |
| `providers/bedrock.py` | Converse client: one account, region and model per call; bounded retries; no failover |
| `executor.py`, `store.py`, `ledger.py` | Resumable execution, request-hash store, spend reservation |
| `bank.py`, `design.py` | Prompt-bank schema and validation; experiment builders; forecast |
| `lexicon_v2.py`, `scoring.py` | Layered span lexicon and lexical baselines (incl. the exact May 2026 GSS) |
| `remedies.py` | Six published inference-time remedies (verbatim prompts) and the proposed RGR |
| `judges.py` | Rubric judge prompt, parsing, other-family assignment, third vote, consensus |
| `gates.py` | protocol_frozen and evaluator_validated gates |
| `annotation/` | Label schema, sheets, sampling, reliability, annotator pack builder |
| `analysis/` | Agreement, cluster bootstrap, estimands, evaluator audit, power, GEE, salvage |
| `data/` | `prompt_bank.csv`, `dialogue_bank.csv`; `drafts/` holds the draft generator |
| `manifests/` | Prices, gates, ledger, run manifest, evaluator audit. No secrets |
| `outputs/` | Generations, judgments, scores, analysis (paid runs) |

Account keys are read from `Codes/.env` (`AWS_ACCESS_KEY`/`AWS_SECRET_KEY` as `aws_a`,
`AWS_ACCESS_KEY2`/`AWS_SECRET_KEY2` as `aws_b`) and never written anywhere.
