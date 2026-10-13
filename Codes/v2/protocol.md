# Study protocol (v2): When Identity Belongs in the Answer

**Status:** FROZEN on 9 October 2026, after the pilot, the two-person bank review and the
development runs, and before any locked-split run (`manifests/gates.json`). The freeze records
SHA-256 hashes of this file, both prompt banks, `config_v2.py`, `manifests/prices.json`,
`remedies.py`, `judges.py`, `lexicon_v2.py` and `annotation/schema.py`. Locked runs refuse to start
if any of them changes; any later change must be logged in Section 10 and re-frozen.

**Supersedes:** `Codes/pre_registration.md` (May 2026 design), which is archived and must not be
described as a preregistration of this study. The May 2026 outputs are an exploratory archive
(`python run_all.py salvage` reconstructs them after deduplication).

**Source plan:** `Future_Plan.md` (revised 8 October 2026). Section numbers below refer to it.

## 1. Research questions (s3)

- **RQ1, measurement:** can a context-aware evaluator distinguish unnecessary from task-required
  identity use on held-out human labels, beyond lexical density and prompt-overlap baselines?
- **RQ2, dialogue:** does irrelevant identity-bearing history change identity use on a later task,
  compared with a matched neutral history answering the same task?
- **RQ3, remedy:** does Relevance-Gated Revision (RGR, `remedies.py`) reduce unnecessary identity
  use without unacceptable loss of adequacy or required identity information, compared with
  ordinary generation (B0), blanket suppression (B1), a relevance instruction (B2) and six
  published inference-time remedies (Si et al. 2023; Ganguli et al. 2023; Panda et al. 2025;
  Gallegos et al. 2025; Bai et al. 2022; Furniturewala et al. 2024)?

## 2. Unit and labels (s4)

The unit is one response with its current request and full preceding conversation. Labels and
allowed values are defined once in `annotation/schema.py` and shared by the human sheets and the
judges. **Primary outcome:** any unnecessary identity mention (the least relevant mention is
rated `unnecessary`). Unsupported personal attribution is reported separately. Adequacy (0–2) and
retention of required identity information are the intervention safeguards.

## 3. Data (s5.1)

| Set | Families | Use |
|---|---:|---|
| Pilot | 24 (6 per category) | Settings and rubric refinement; never in confirmatory analysis |
| Development | 80 | Prompt, threshold and judge-example selection |
| Locked | 160 | Confirmatory single-turn and intervention estimates |
| Dialogue | 80 | Confirmatory dialogue contrast |

Each core family has three conditions: identity absent, incidental and required. Splits and the
intervention (40 dev + 80 locked), paraphrase (60) and schema (60) subsets are assigned by
`python run_all.py assign` with seed 20261008, balanced by category, before outputs are viewed.

## 4. Generation (s5.2, s5.3, s8)

| Setting | Value |
|---|---|
| Region / account | us-east-1 / `aws_a` (primary mode; no failover) |
| Generators | Nova Lite, Nova Pro, Gemma 3 12B, Qwen3 32B, Mistral Large 3, DeepSeek V3.2 |
| Temperature / output cap | 0.7 / 384 tokens (see open decision D1) |
| Draws | 2 per condition (Bedrock Converse exposes no seed; draws are repeated calls) |
| System prompts | B0, B1, B2 and the May 2026 JSON schema, verbatim in `config_v2.py` |
| Dialogue | identity vs neutral opening, 2 trajectories each, opening + 3 identical switched tasks, real replies fed back |

## 5. Judges (s4)

Two other-family judges per output (Nova Pro and Mistral Large 3; DeepSeek V3.2 substitutes for a
same-family judge). Third vote by the first eligible of Qwen3 32B, DeepSeek V3.2, ... only on
disagreement or abstention, capped at 10% of judged outputs. Judges see the conversation, current
request and response only. Prompt version `judge-v1-2026-10-08` (`judges.py`).

## 6. Human validation (s6)

Three independent raters, 720 units each: pilot 60, representative 360 (240 single-turn + 120
dialogue first-switch, locked, model x category balanced, inclusion probabilities recorded),
challenge 120 (enriched, never used for prevalence), intervention review 180 (60 matched
B0/B1/B2 triplets). Guidelines: `annotation/guidelines.md` (pack version 1.0).

**Gates (decision criteria, not guarantees):** pilot alpha on the primary outcome of at least
0.67 (interpreted with prevalence); evaluator precision of at least 0.80 and recall of at least 0.70
on held-out labels with at least 50 positive and 50 negative cases; demonstrated F1 gain over the
best dev-selected baseline, or equal performance at lower cost; intervention target of 10 points
with at most 5 points loss of adequacy or retention (one-sided 95% bound).

## 7. Analysis (s7)

Estimands are equal-weight averages over model x category cells on locked families. Intervals
come from a stratified family-cluster bootstrap (2,000 replicates; all variants, draws, models,
arms and turns of a family move together). Primary contrast family with Holm correction:
evaluator gain over the best baseline (directional), first-switch dialogue effect, and RGR vs B0
unnecessary-mention prevalence on absent and incidental conditions. Secondary: every remedy vs
B0, RGR vs every other remedy, B2 vs B1; non-inferiority of retention and adequacy for every
remedy at a 5-point margin. Later turns, category and model breakdowns are secondary. A
logistic GEE with exchangeable within-family correlation is a secondary sensitivity model.

## 7a. Remedies (added 8 October 2026)

Published comparators run with the verbatim prompts in `remedies.py` on the 80 locked
intervention-subset families; adaptations from multiple-choice to open-ended answers are marked
ADAPTED in the code. RGR runs on the 40 dev and 80 locked intervention-subset families; its gate
and revision prompts are frozen after dev runs. Every multi-step method reuses the same B0 draft.
The human intervention review uses 45 matched sets of B0, B1, B2 and RGR (180 units). The budget
cap was raised from US$300 to US$450 for these runs (forecast: US$287 inference at the planning
allowance, about US$165 at listed tariffs).

## 8. Pilot record (8 October 2026)

| Item | Observation |
|---|---|
| Smoke test | 6 of 6 generators answered on `aws_a` with the system field |
| Pilot generation | 288 of 288 calls succeeded (Nova Lite, DeepSeek V3.2); ledger US$0.15 at budget rate, US$0.05 at listed tariffs |
| Tokens | Mean input 53–57 and output 90–145 tokens per call, below the 250/220 planning averages |
| Truncation | DeepSeek V3.2 stopped at the 384-token cap on 20 of 144 calls; Nova Lite on 0 |
| Prompt bank | The 24 pilot families are AI-assisted drafts that have not yet had the two human reviews |
| Annotation | Pilot batch of 60 units (30 per model, 15 per category, 20 per condition) issued as annotator pack v1.0 |

## 8a. Pilot annotation and bank review (9 October 2026)

| Item | Result |
|---|---|
| Pilot annotation | Three raters (R1-R3), 60 units each, complete. Primary-label Krippendorff alpha 0.870 (target 0.67); raw agreement 0.989. Task adequacy alpha 0.483 (ordinal) with 91% of labels at 2. |
| Pilot prevalence | Unnecessary mention on 3/60 units (adjudicated), all in the incidental condition (3/20); 0/20 absent; 0/20 required. |
| Adjudication | 19 field-level disagreements; all resolved by 2-of-3 majority; first author confirmed each; machine-drafted rationales were available to the adjudicator and are not used as labels. |
| Bank review | Two reviewers (A, B), all 720 core rows and 80 dialogue families. Raw agreement on core checks 0.969-0.999; dialogue matching agreement 0.425 (A flagged 48 families, B flagged 2). 38 core and 48 dialogue disagreements adjudicated by the first author. 63 edits applied (`data/review/applied_edits.csv`). |
| Correction | COR-C052 incidental: the adjudicated edit removed the identity detail and duplicated the absent row; reviewer B's matched edit was applied instead. Confirmed by the first author on 9 Oct 2026. |
| Lexicon | v2.1: general lists of Indian caste and community names (context rule against surname use), ethnic and tribal groups, national demonyms, hyphenated compounds and more person nouns. Gap found by checking the bank, so locked prompts were visible while extending; lists are general, not bank-derived. Incidental-row coverage 186/240 -> 240/240; absent-row group-label hits 0 -> 1 (a nationality used as a place). |

## 8b. Development runs (9 October 2026)

All dev-split experiments ran at the 1,024-token cap with balanced accounts: 2,880 single-turn,
2,880 B1/B2, 1,440 RGR finals (1,475 extra calls), 756 paraphrase and 228 schema answers, with no
unresolved failures. Truncation at the cap was at most 2.3% per model; Bedrock content filtering
stopped 1.0-1.5% of Nova answers. RGR gate on dev: 30% skipped (no identity terms), 37% unchanged
after the gate, 32% revised, 0.3% reverted by the retention guard; 1.02 extra calls per answer.
RGR prompts were not changed after the dev runs.

## 9. Decisions taken before freezing

- **D1, output cap. DECIDED 9 Oct (revised):** 384 -> 512 after the pilot; dev runs at 512 still truncated DeepSeek V3.2 (17%) and Qwen3 (11%) on planning/checklist tasks, so the cap is 1,024. The 512-token dev outputs are archived in `outputs/archive/` and excluded from analysis.
- **D2a, accounts. DECIDED 9 Oct:** `balanced` mode; families alternate between `aws_a` and `aws_b` within category (both verified for all six models). Per-model concurrency limits (Nova Pro 2, Nova Lite 4, DeepSeek 3) and up to 8 bounded retries after Nova Pro throttling.
- **D5, intervention target. DECIDED 9 Oct:** a 50% relative reduction of unnecessary mentions on
  non-required conditions replaces the absolute 10-point target (power about 0.96 for 80 locked
  families at the pilot rate of 7.5%; about 0.40 for a 25% reduction).
- **D6, evaluator positives.** At the pilot rate, the 360 representative units yield roughly 25
  human positives, short of the 50 needed; the challenge sample must be weighted towards
  rare positives, or the positive target restated, before freezing.
- **D2, primary account.** Confirm credit eligibility, balance and quotas for `aws_a`.
- **D3, prices.** Confirm Nova and Qwen3 32B tariffs on the live pricing page and set
  `verified: true` in `manifests/prices.json`.
- **D4, banks.** Drafted on 8 October (`data/prompt_bank.csv`, `data/dialogue_bank.csv`); every row still needs two human reviews (`review_status` = `reviewed_2`).

## 10. Deviations log

| Date | Deviation | Reason |
|---|---|---|
| 2026-10-08 | Pilot run on AI-drafted, unreviewed pilot families | Pilot purpose is settings and rubric refinement; pilot families are excluded from confirmatory analysis |
| 2026-10-08 | Primary remedy contrast changed from B2 vs B1 to RGR vs B0; six published comparators added | Before any dev or locked run; protocol not yet frozen |
| 2026-10-08 | Core (240) and dialogue (80) families drafted with AI assistance; 60 paraphrases written independently | Two human reviews required before freezing (review_status) |
| 2026-10-09 | Output cap 384 -> 512; intervention target restated as 50% relative reduction; lexicon v2.1 | Pilot evidence; before dev runs |
| 2026-10-09 | Output cap 512 -> 1,024; dev runs repeated; balanced accounts; concurrency limits | Dev truncation and throttling; before freezing |
| 2026-10-09 | Paraphrases of COR-R047 and COR-R048 (required) rewritten after adjudicated prompt edits | Not independently reviewed |
