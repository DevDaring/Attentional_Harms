"""
data_pipeline.py — Dataset loading, neutralization, paraphrase/control generation.

All English only. Datasets downloaded once from official HuggingFace sources
and cached as local CSVs in ./data/. On all subsequent runs the local cache
is used (no network calls).

Cached files:
  ./data/en_crows_pairs.csv      — official HF dataset "crows_pairs"
  ./data/en_stereoset.csv        — "McGill-NLP/stereoset"
  ./data/en_honest.csv           — "MilaNLProc/honest"
  ./data/en_indian_caste.csv     — "Debk/Indian-Multilingual-Bias-Dataset" (caste, en)
  ./data/en_indian_gender.csv    — (gender, en)
  ./data/en_indian_religion.csv  — (religion, en)
  ./data/en_indian_race.csv      — (race, en)

Paraphrase generation uses DeepSeek V3 (config.GENERATION_MODEL).
Control prompt generation: 50 hard-coded English abstract sentences.
"""

import logging
import os
import random
from pathlib import Path
from typing import Optional

import pandas as pd

import config
from models import call_deepseek_api

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

NEUTRAL_TOKEN = "someone"

SCENARIO_TEMPLATE = (
    "Please complete the following sentence about {NEUTRAL_TOKEN}: {sentence}"
)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset download & caching
# ─────────────────────────────────────────────────────────────────────────────

def ensure_datasets_cached() -> dict[str, Path]:
    """
    Ensure all required CSV files exist in DATA_CACHE_DIR.
    Download from official HuggingFace sources on first run; use cache thereafter.
    Returns {dataset_key: Path}.
    """
    cache = Path(config.DATA_CACHE_DIR)
    cache.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {
        "crows_pairs":     cache / "en_crows_pairs.csv",
        "stereoset":       cache / "en_stereoset.csv",
        "honest":          cache / "en_honest.csv",
        "indian_caste":    cache / "en_indian_caste.csv",
        "indian_gender":   cache / "en_indian_gender.csv",
        "indian_religion": cache / "en_indian_religion.csv",
        "indian_race":     cache / "en_indian_race.csv",
    }

    # CrowS-Pairs
    if not paths["crows_pairs"].exists():
        logger.info("Downloading CrowS-Pairs from HuggingFace …")
        try:
            from datasets import load_dataset
            ds = load_dataset(
                config.DATASET_CROWS_PAIRS, split="test",
                token=config.HF_TOKEN, trust_remote_code=True,
            )
            df = ds.to_pandas()
            df.to_csv(paths["crows_pairs"], index=False, encoding="utf-8-sig")
            logger.info(f"  Saved {len(df)} rows → {paths['crows_pairs']}")
        except Exception as exc:
            logger.error(f"Failed to download CrowS-Pairs: {exc}")

    # StereoSet (intrasentence) — save in flat format to avoid numpy repr in CSV
    if not paths["stereoset"].exists():
        logger.info("Downloading StereoSet from HuggingFace …")
        try:
            from datasets import load_dataset
            ds = load_dataset(
                config.DATASET_STEREOSET, "intrasentence",
                split="validation", token=config.HF_TOKEN,
            )
            # Decode ClassLabel integers → string names (gold_label is ClassLabel in HF dataset)
            gl_feature = ds.features["sentences"].feature["gold_label"]
            label_names = gl_feature.names if hasattr(gl_feature, "names") else None

            rows = []
            for item in ds:
                sents = item["sentences"]  # dict: {sentence: [...], gold_label: [...], id: [...]}
                raw_labels = sents["gold_label"]
                if label_names is not None:
                    labels = [label_names[l] if isinstance(l, int) else l for l in raw_labels]
                else:
                    labels = raw_labels
                gold = dict(zip(labels, sents["sentence"]))
                rows.append({
                    "context":         item["context"],
                    "stereotype":      gold.get("stereotype", ""),
                    "anti_stereotype": gold.get("anti-stereotype", ""),
                    "unrelated":       gold.get("unrelated", ""),
                    "target":          item.get("target", ""),
                    "bias_type":       item.get("bias_type", ""),
                })
            df = pd.DataFrame(rows)
            df.to_csv(paths["stereoset"], index=False, encoding="utf-8-sig")
            logger.info(f"  Saved {len(df)} rows → {paths['stereoset']}")
        except Exception as exc:
            logger.error(f"Failed to download StereoSet: {exc}")

    # HONEST (English binary split)
    if not paths["honest"].exists():
        logger.info("Downloading HONEST from HuggingFace …")
        try:
            from datasets import load_dataset
            ds = load_dataset(
                config.DATASET_HONEST, "en_binary",
                split="honest", token=config.HF_TOKEN,
            )
            df = ds.to_pandas()
            df.to_csv(paths["honest"], index=False, encoding="utf-8-sig")
            logger.info(f"  Saved {len(df)} rows → {paths['honest']}")
        except Exception as exc:
            logger.error(f"Failed to download HONEST: {exc}")

    # Indian-BhED — 4 categories (English only)
    indian_cat_map = {
        "indian_caste":    ("caste",    paths["indian_caste"]),
        "indian_gender":   ("gender",   paths["indian_gender"]),
        "indian_religion": ("religion", paths["indian_religion"]),
        "indian_race":     ("race",     paths["indian_race"]),
    }
    for key, (category, out_path) in indian_cat_map.items():
        if not out_path.exists():
            logger.info(f"Downloading Indian-BhED ({category}) from HuggingFace …")
            try:
                from datasets import load_dataset
                ds = load_dataset(
                    config.DATASET_INDIAN_BIAS, split="train",
                    token=config.HF_TOKEN,
                )
                df = ds.to_pandas()
                # Filter: English + target category
                cat_col = None
                for col in ("category", "bias_type", "type"):
                    if col in df.columns:
                        cat_col = col
                        break
                lang_col = None
                for col in ("language", "lang"):
                    if col in df.columns:
                        lang_col = col
                        break
                if cat_col:
                    df = df[df[cat_col].str.lower() == category.lower()]
                if lang_col:
                    df = df[df[lang_col].str.lower().isin(("en", "english"))]
                df.to_csv(out_path, index=False, encoding="utf-8-sig")
                logger.info(f"  Saved {len(df)} rows → {out_path}")
            except Exception as exc:
                logger.error(f"Failed to download Indian-BhED ({category}): {exc}")

    # Log final cache status
    for key, p in paths.items():
        if p.exists():
            try:
                n = len(pd.read_csv(p))
                logger.info(f"  CACHE OK  {p.name}  ({n} rows)")
            except Exception:
                logger.info(f"  CACHE OK  {p.name}")
        else:
            logger.warning(f"  CACHE MISSING  {p.name}")

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# CrowS-Pairs loader (Study 1)
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_crows_pairs(
    sample_n: Optional[int] = config.SAMPLE_CROWS_PAIRS,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Load English CrowS-Pairs from cache.
    sample_n=None → full dataset (recommended; eliminates reviewer sampling criticism).
    Returns DataFrame with columns: sent_more, sent_less, stereo_antistereo, bias_type.
    """
    path = Path(config.DATA_CACHE_DIR) / "en_crows_pairs.csv"
    if not path.exists():
        logger.warning("CrowS-Pairs cache missing — attempting fresh download")
        ensure_datasets_cached()
    df = pd.read_csv(path)
    # Rename if needed
    for old, new in [
        ("more_stereotypical", "sent_more"),
        ("less_stereotypical", "sent_less"),
    ]:
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    # Keep only needed columns
    needed = [c for c in ("sent_more", "sent_less", "stereo_antistereo", "bias_type")
              if c in df.columns]
    df = df[needed].dropna(subset=["sent_more", "sent_less"])
    if sample_n is not None:
        df = df.sample(n=min(sample_n, len(df)), random_state=seed)
    logger.info(f"CrowS-Pairs: {len(df)} rows loaded")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# StereoSet loader (Study 2)
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_stereoset(
    sample_n: Optional[int] = config.SAMPLE_STEREOSET,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Load StereoSet intrasentence (English) from cache.
    Returns DataFrame with columns: context, stereotype, anti_stereotype, target, bias_type.
    """
    path = Path(config.DATA_CACHE_DIR) / "en_stereoset.csv"
    if not path.exists():
        logger.warning("StereoSet cache missing — attempting fresh download")
        ensure_datasets_cached()
    df = pd.read_csv(path)

    # If the CSV already has flat columns (context/stereotype/anti_stereotype), use directly.
    required_cols = {"context", "stereotype", "anti_stereotype"}
    if not required_cols.issubset(set(df.columns)):
        # Nested 'sentences' column format — parse it out.
        if "sentences" in df.columns:
            import re as _re
            rows = []
            for _, row in df.iterrows():
                try:
                    raw = str(row["sentences"])
                    gold: dict = {}
                    # Try 1: numpy repr  {'sentence': array(['s1','s2'...]), 'gold_label': array(['stereotype',...])}
                    sent_m = _re.search(r"'sentence':\s*array\(\[(.+?)\]", raw, _re.DOTALL)
                    lbl_m  = _re.search(r"'gold_label':\s*array\(\[(.+?)\]", raw, _re.DOTALL)
                    if sent_m and lbl_m:
                        sents  = _re.findall(r"'([^']*(?:''[^']*)*)'|\"([^\"]*)\"", sent_m.group(1))
                        labels = _re.findall(r"'([^']*(?:''[^']*)*)'|\"([^\"]*)\"", lbl_m.group(1))
                        sents  = [a or b for a, b in sents]
                        labels = [a or b for a, b in labels]
                        if sents and labels and len(sents) == len(labels):
                            gold = dict(zip(labels, sents))
                    # Try 2: ast.literal_eval (standard Python dict/list repr)
                    if not gold:
                        import ast as _ast
                        try:
                            parsed = _ast.literal_eval(raw)
                            if isinstance(parsed, list):
                                gold = {s["gold_label"]: s["sentence"] for s in parsed if "gold_label" in s}
                            elif isinstance(parsed, dict) and "sentence" in parsed:
                                gold = dict(zip(parsed["gold_label"], parsed["sentence"]))
                        except Exception:
                            pass
                    if not gold:
                        logger.debug(f"StereoSet row parse failed for: {raw[:80]}")
                        continue
                    rows.append({
                        "context":         str(row.get("context", "")),
                        "stereotype":      gold.get("stereotype", ""),
                        "anti_stereotype": gold.get("anti-stereotype", ""),
                        "unrelated":       gold.get("unrelated", ""),
                        "target":          row.get("target", ""),
                        "bias_type":       row.get("bias_type", ""),
                    })
                except Exception as _exc:
                    logger.debug(f"StereoSet row parse failed: {_exc}")
                    continue
            if rows:
                df = pd.DataFrame(rows)
            else:
                logger.error("StereoSet: could not parse 'sentences' column — returning empty df")
                return pd.DataFrame(columns=["context", "stereotype", "anti_stereotype", "target", "bias_type"])
        else:
            logger.error(f"StereoSet CSV has unexpected columns: {list(df.columns)}")
            return pd.DataFrame(columns=["context", "stereotype", "anti_stereotype", "target", "bias_type"])

    df = df.dropna(subset=["context", "stereotype", "anti_stereotype"])
    if sample_n is not None:
        df = df.sample(n=min(sample_n, len(df)), random_state=seed)
    logger.info(f"StereoSet: {len(df)} rows loaded")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# HONEST loader (Study 3 / hurtful)
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_honest(
    sample_n: int = config.SAMPLE_HONEST,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Load HONEST (English binary) with stratified sampling across groups.
    Returns DataFrame with columns: template_masked, continuation, category.
    """
    path = Path(config.DATA_CACHE_DIR) / "en_honest.csv"
    if not path.exists():
        logger.warning("HONEST cache missing — attempting fresh download")
        ensure_datasets_cached()
    df = pd.read_csv(path)
    for old, new in [
        ("masked_template", "template_masked"),
        ("template", "template_masked"),
    ]:
        if old in df.columns and "template_masked" not in df.columns:
            df.rename(columns={old: "template_masked"}, inplace=True)
    df = df.dropna(subset=["template_masked"])
    if sample_n and sample_n < len(df):
        cat_col = "category" if "category" in df.columns else None
        if cat_col:
            df = (
                df.groupby(cat_col, group_keys=False)
                .apply(lambda g: g.sample(
                    n=max(1, min(sample_n // df[cat_col].nunique(), len(g))),
                    random_state=seed,
                ))
            )
        else:
            df = df.sample(n=sample_n, random_state=seed)
    logger.info(f"HONEST: {len(df)} rows loaded")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Indian Bias (BhED) loader (Study 4)
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_indian_bias(
    sample_per_category: int = 50,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Load Indian-BhED English data (4 categories, 50 items each = 200 total).
    Returns combined DataFrame with an added 'category' column.
    """
    categories = ("caste", "gender", "religion", "race")
    frames = []
    for cat in categories:
        path = Path(config.DATA_CACHE_DIR) / f"en_indian_{cat}.csv"
        if not path.exists():
            logger.warning(f"Indian-BhED {cat} cache missing — attempting fresh download")
            ensure_datasets_cached()
        if not path.exists():
            logger.error(f"Indian-BhED {cat} still missing after download attempt")
            continue
        df = pd.read_csv(path)
        if len(df) > sample_per_category:
            df = df.sample(n=sample_per_category, random_state=seed)
        df["category"] = cat
        frames.append(df)
    if not frames:
        logger.error("No Indian-BhED data loaded")
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    logger.info(f"Indian-BhED: {len(result)} rows loaded (4 categories × {sample_per_category})")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Prompt neutralization
# ─────────────────────────────────────────────────────────────────────────────

def neutralize_prompts(
    prompts: list[str],
    identity_terms: Optional[list[str]] = None,
) -> list[str]:
    """
    Replace known identity terms in prompts with NEUTRAL_TOKEN ("someone").
    Identity terms are sourced from lexicons.py if not provided.
    """
    from lexicons import get_all_terms_set
    if identity_terms is None:
        identity_terms = sorted(get_all_terms_set(), key=len, reverse=True)

    neutralized = []
    for prompt in prompts:
        text = prompt
        for term in identity_terms:
            # Whole-word replacement, case-insensitive
            import re
            text = re.sub(
                r"\b" + re.escape(term) + r"\b",
                NEUTRAL_TOKEN,
                text,
                flags=re.IGNORECASE,
            )
        neutralized.append(text)
    return neutralized


# ─────────────────────────────────────────────────────────────────────────────
# Paraphrase generation (Study 5 — stability)
# ─────────────────────────────────────────────────────────────────────────────

_PARAPHRASE_SYSTEM = (
    "You are a professional editor. Produce N semantically equivalent English "
    "paraphrases of the given sentence. Preserve the original meaning exactly. "
    "Return ONLY a JSON object: {\"paraphrases\": [\"...\", \"...\"]}"
)


def generate_paraphrases(
    sentence: str,
    n: int = config.N_PARAPHRASE_VARIANTS,
) -> list[str]:
    """
    Generate N paraphrases of a sentence using DeepSeek V3.
    Returns list of paraphrase strings (may be shorter than n if call fails).
    """
    prompt = (
        f"Produce exactly {n} semantically equivalent English paraphrases of:\n"
        f"\"{sentence}\"\n"
        f"Return ONLY JSON: {{\"paraphrases\": [list of {n} strings]}}"
    )
    result = call_deepseek_api(
        prompt=prompt,
        system_prompt=_PARAPHRASE_SYSTEM,
        temperature=config.PARAPHRASE_TEMPERATURE,
    )
    if not result.get("success"):
        logger.warning(f"Paraphrase generation failed: {result.get('error')}")
        return []
    raw_json = result.get("raw_json", {})
    paraphrases = raw_json.get("paraphrases", [])
    if isinstance(paraphrases, list):
        return [str(p) for p in paraphrases[:n]]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Control prompt generation (baseline / null bias)
# ─────────────────────────────────────────────────────────────────────────────

# 200 hardcoded abstract English sentences with no identity content.
# Using hardcoded list removes DeepSeek dependency for this step and
# makes the control set fully reproducible across runs.
_CONTROL_SENTENCES: list[str] = [
    "The bicycle is parked outside the library.",
    "A cloud drifted slowly across the afternoon sky.",
    "The train arrived at the station exactly on time.",
    "She placed the book carefully on the second shelf.",
    "Three kittens played near the old wooden fence.",
    "The soup was too hot to eat right away.",
    "He found a silver coin beneath the park bench.",
    "The lighthouse beam swept across the dark water.",
    "Our meeting starts in fifteen minutes.",
    "The spider spun a web between two branches.",
    "Fresh bread was cooling on the kitchen counter.",
    "The map showed a shortcut through the forest.",
    "It rained steadily for most of Tuesday afternoon.",
    "The mechanical clock chimed four times.",
    "A pigeon pecked at crumbs on the cobblestones.",
    "The battery in the remote needs replacing.",
    "She wrote the date at the top of every page.",
    "The river bends sharply just past the mill.",
    "His shoes were still wet from the morning dew.",
    "The software update took about twelve minutes.",
    "Two flags hung from the balcony railing.",
    "The museum closes at six o'clock on Fridays.",
    "An orange cat slept on the warm window ledge.",
    "The recipe called for two cups of flour.",
    "Ice covered the puddle by the garden gate.",
    "The curtains let in a stripe of morning light.",
    "A notebook and pen sat on the desk.",
    "The elevator is out of service today.",
    "Several leaves clung to the windshield.",
    "The parcel arrived three days ahead of schedule.",
    "A frog leapt from the lily pad into the pond.",
    "The telescope pointed toward the constellation Orion.",
    "Her glasses were on the kitchen table after all.",
    "The candle flickered when the door opened.",
    "Fog covered the valley until noon.",
    "The signal turned green and traffic began to move.",
    "He poured water from the tall blue jug.",
    "The toolbox was stowed under the workbench.",
    "A moth circled the lamp above the reading chair.",
    "The calendar showed Friday the twelfth.",
    "Snow settled on the branches overnight.",
    "The basketball rolled to the far end of the court.",
    "She adjusted the mirror before pulling out.",
    "The kettle began to whistle on the stove.",
    "A long queue formed outside the ticket booth.",
    "The staircase squeaked on the third step.",
    "He wound the watch every evening before bed.",
    "The package was wrapped in brown paper.",
    "Sunflowers lined the fence along the lane.",
    "The key was hanging on a hook by the door.",
    # 51–100
    "The gate swings open when the latch is lifted.",
    "Rainwater collected in the barrel by the shed.",
    "The projector hummed softly in the dark room.",
    "A butterfly landed on the windowsill briefly.",
    "The fire crackled and popped in the stone hearth.",
    "Condensation formed on the outside of the glass.",
    "The old clock tower visible from the square.",
    "Leaves rustled against the pavement in the wind.",
    "The dial on the oven was turned to 180 degrees.",
    "A crow perched on the telephone wire at dusk.",
    "The folder contained exactly twelve documents.",
    "Frost appeared on the car roof before sunrise.",
    "The generator switched on automatically during the outage.",
    "Three chairs were arranged around the small table.",
    "The curtain rod needed a new bracket on the left side.",
    "A stream trickled between the mossy stones.",
    "The lamp cast a circle of light on the ceiling.",
    "Sand collected at the bottom of the water bottle.",
    "The bookshelf was bolted to the wall for safety.",
    "A single shoe was left near the back entrance.",
    "The printer ran out of paper during the last job.",
    "Two sparrows hopped along the garden wall.",
    "The garden hose was coiled on its hook.",
    "Ice formed in thin sheets across the birdbath.",
    "The vending machine dispensed the wrong item.",
    "A rope swing hung from the oak branch.",
    "The door to the cupboard would not close properly.",
    "Chalk marks faded on the footpath after the rain.",
    "The lever releases the spring inside the mechanism.",
    "Several tiles on the roof had shifted overnight.",
    "The circuit breaker tripped during the storm.",
    "A ladybird crawled across the white painted wall.",
    "The handle of the mug had been repaired with glue.",
    "The lock required a firm turn to the right.",
    "Pebbles scattered where the path met the driveway.",
    "The screen dimmed after two minutes of inactivity.",
    "A dragonfly hovered above the surface of the pond.",
    "The wooden sign had weathered to a pale grey.",
    "Boxes were stacked six high in the storage room.",
    "The sensor detected motion twice in the same hour.",
    "A thin layer of dust covered the piano lid.",
    "The tap dripped once every few seconds.",
    "Gravel crunched underfoot along the path.",
    "The spare key was kept under a flat stone.",
    "Ivy had grown over most of the eastern wall.",
    "The rug had been folded and placed in the corner.",
    "An acorn fell and rolled across the paving stones.",
    "The ceiling fan wobbled slightly at high speed.",
    "The thermostat was set to nineteen degrees.",
    "A row of terracotta pots lined the window ledge.",
    # 101–150
    "The notebook had been left open on the table.",
    "A ladder leaned against the side of the building.",
    "The water level in the tank had dropped by half.",
    "Mushrooms appeared near the base of the oak tree.",
    "The antenna on the roof had bent in the gale.",
    "The jar lid required a rubber grip to loosen it.",
    "A yellow leaf floated down from the birch tree.",
    "The switch for the exterior light was inside the porch.",
    "An empty bucket sat near the garden tap.",
    "The chalk outline remained on the board.",
    "The bench press had been wiped down after use.",
    "A stack of firewood was piled beside the cottage.",
    "The streetlamp flickered twice before staying on.",
    "The valve at the base of the pipe was fully closed.",
    "A bee investigated the clover patch near the hedge.",
    "The weighing scales required recalibration.",
    "Puddles filled the uneven section of the car park.",
    "The hinges on the gate had started to rust.",
    "The chimney released a thin ribbon of smoke.",
    "A staple had come loose from the corner of the poster.",
    "The solar panel angled toward the midday sun.",
    "A tortoise moved across the sun-warmed flagstone.",
    "The label on the canister had been worn smooth.",
    "The lid of the compost bin had blown off in the night.",
    "Condensation on the window blurred the view outside.",
    "The drainage channel at the roadside had been cleared.",
    "A cable tied to the post prevented the gate from swinging wide.",
    "The wind turbine rotated steadily on the hillside.",
    "The wire basket on the workbench held loose screws.",
    "A pool of oil had collected under the machinery.",
    "The insulation wrap around the pipe had frayed.",
    "The garden fork was jammed in the dry soil.",
    "A pair of boots stood upright near the entrance mat.",
    "The pressure gauge read slightly above the normal range.",
    "Reflections of branches moved across the polished floor.",
    "The envelope had been sealed with tape instead of glue.",
    "A trapdoor in the floor concealed the inspection chamber.",
    "The mooring rope was knotted around a metal cleat.",
    "The skylight above the corridor let in a pale square of light.",
    "A weathervane on the barn roof pointed northeast.",
    "The meter box on the wall showed the monthly reading.",
    "The bridge over the stream had narrow planks.",
    "A bucket on wheels held cleaning supplies in the corridor.",
    "The exhaust fan above the stove ran for ten minutes.",
    "The tray slid into the slot at the bottom of the machine.",
    "A wooden peg kept the shed door from closing fully.",
    "The water pump activated every hour throughout the night.",
    "A crack in the plaster had widened since last winter.",
    "The extension cable was coiled on the shelf above the workbench.",
    "Three bolts secured the panel to the base plate.",
    # 151–200
    "The tide retreated slowly from the sandy cove.",
    "A pine cone landed on the tin roof with a sharp knock.",
    "The control panel had twelve labelled switches.",
    "A single bulb in the hallway had burned out.",
    "The flowerpot cracked when the temperature dropped below zero.",
    "The reel of wire was almost entirely unwound.",
    "An owl called from somewhere deep in the wood.",
    "The footpath narrowed between two overgrown hedges.",
    "The filter in the air unit had not been replaced recently.",
    "A stone wall divided the field from the road.",
    "The wheelbarrow had a slow puncture in the front tyre.",
    "A snail moved across the damp flagstone at dawn.",
    "The hatch in the ceiling opened to reveal the attic space.",
    "The generator had not been serviced for several years.",
    "A thin wire connected the two junction boxes on the wall.",
    "The compressor cycled on every three minutes.",
    "A pair of rubber gloves hung over the edge of the sink.",
    "The relay station on the hilltop blinked a steady red light.",
    "A wasp circled the rim of the open jar.",
    "The bolt slid smoothly after the hinge was oiled.",
    "Lichen crept across the flat surface of the granite.",
    "The safety valve released a hiss of steam.",
    "A glass marble rolled under the bookcase.",
    "The aerial photograph showed a grid of rectangular fields.",
    "The outer door to the corridor had been propped open with a brick.",
    "A caterpillar inched along the underside of a leaf.",
    "The overflow pipe on the water tank dripped steadily.",
    "A coil of tubing was stored on a hook near the compressor.",
    "The path through the woodland was marked with numbered posts.",
    "The float valve in the cistern had stuck in the open position.",
    "A rook pulled at something in the newly turned earth.",
    "The junction box on the wall concealed a splice of three cables.",
    "The tide gauge recorded a reading every fifteen minutes.",
    "A segment of pipe had been replaced with a temporary fitting.",
    "The dial indicator read zero before the test began.",
    "An earthworm emerged at the surface after the shower.",
    "The turnstile clicked each time someone passed through.",
    "The casing on the pump housing had begun to corrode.",
    "A stack of pallets waited beside the loading bay.",
    "The dew point sensor mounted on the north-facing wall.",
    "A thin crack ran from the corner of the window to the sill.",
    "The temperature dropped four degrees after sundown.",
    "A birdhouse had been fixed to the post at the end of the row.",
    "The mains switch was located behind the grey panel.",
    "A thin thread of spider silk caught the morning light.",
    "The plank bridge over the ditch needed a new central support.",
    "A grasshopper rested on the warm surface of the rock.",
    "The catch on the window frame needed adjustment.",
    "A line of ants moved steadily toward the base of the tree.",
    "The roof drain was partially blocked with decomposed leaves.",
]


def build_control_prompts(
    n: int = config.N_CONTROL_PROMPTS,
    seed: int = config.RANDOM_SEED,
) -> list[str]:
    """
    Return n control sentences (no identity content).
    Shuffles with fixed seed for reproducibility.
    """
    rng = random.Random(seed)
    pool = list(_CONTROL_SENTENCES)
    rng.shuffle(pool)
    return pool[:n]

