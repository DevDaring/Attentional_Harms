"""
lexicons.py — English identity-term lexicons for GSS / FI / VAR computation.

English-only. All Hindi/Bengali terms have been permanently removed.

Four dimensions: caste, gender, religion, race_ethnicity.
Sources:
  caste         — Samara et al. 2023; BhED annotations (Bhatt et al. 2022)
  gender        — WinoBias (Zhao et al. 2018) + manual extension
  religion      — HolisticBias (Smith et al. 2022)
  race_ethnicity— HolisticBias + WinoBias

All entries lower-cased; matching is case-insensitive in lexicon.py consumers.
"""

from typing import Optional

ENGLISH_LEXICONS: dict[str, list[str]] = {

    "caste": [
        # Varna + jati names commonly used in bias literature
        "brahmin", "brahman", "kshatriya", "vaishya", "shudra", "dalit",
        "scheduled caste", "scheduled tribe", "obc", "upper caste", "lower caste",
        "untouchable", "harijan", "bhangi", "chamar", "dhobi", "koli",
        "mahar", "mala", "madiga", "meghwal", "valmiki", "musahar",
        "jatav", "dom", "pariah", "adi dravidar", "oppressed caste",
        "depressed class", "savarna", "avarna", "caste",
    ],

    "gender": [
        # Pronouns & forms
        "he", "she", "they", "him", "her", "his", "hers", "their", "theirs",
        "himself", "herself", "themselves",
        # Role nouns
        "man", "woman", "men", "women", "boy", "girl", "boys", "girls",
        "male", "female", "males", "females",
        # Gendered terms
        "husband", "wife", "son", "daughter", "father", "mother",
        "brother", "sister", "uncle", "aunt", "nephew", "niece",
        "grandfather", "grandmother", "grandson", "granddaughter",
        "king", "queen", "prince", "princess", "mr", "mrs", "ms", "miss",
        # Gender identity
        "transgender", "non-binary", "nonbinary", "enby", "genderqueer",
        "genderfluid", "agender", "intersex", "cisgender", "cis",
    ],

    "religion": [
        # Adherents
        "christian", "muslim", "hindu", "sikh", "jewish", "buddhist",
        "jain", "zoroastrian", "atheist", "agnostic",
        "catholic", "protestant", "evangelical", "orthodox",
        "sunni", "shia", "sufi", "wahhabi",
        # Nouns
        "christianity", "islam", "hinduism", "sikhism", "judaism",
        "buddhism", "jainism", "religion", "faith", "sect",
        # Texts / leaders
        "priest", "pastor", "imam", "rabbi", "monk", "nun", "pandit",
        "maulvi", "granthi", "pope",
        # Places of worship
        "church", "mosque", "temple", "synagogue", "gurdwara", "mandir",
        "mosque", "cathedral", "chapel",
        # Slurs / epithets (included to detect derogatory use)
        "kafir", "infidel", "heretic", "pagan",
    ],

    "race_ethnicity": [
        # Major groupings used in NLP bias literature
        "black", "white", "asian", "hispanic", "latino", "latina",
        "latinx", "african", "african american", "caucasian",
        "native american", "indigenous", "pacific islander",
        "arab", "middle eastern", "south asian", "east asian",
        "southeast asian", "mixed race", "biracial", "multiracial",
        # Nationalities / ethnicities frequently referenced
        "chinese", "indian", "japanese", "korean", "vietnamese",
        "mexican", "puerto rican", "cuban", "jamaican", "nigerian",
        "somali", "ethiopian", "kenyan", "ghanaian",
        # British ethnic monitoring categories
        "british asian", "british black", "british white",
    ],
}

# Flat union of all terms — precomputed for fast lookup
ENGLISH_ALL_TERMS: frozenset[str] = frozenset(
    term.lower() for terms in ENGLISH_LEXICONS.values() for term in terms
)


# ─────────────────────────────────────────────────────────────────────────────
# Public API (language-agnostic signatures — always English)
# ─────────────────────────────────────────────────────────────────────────────

def get_lexicon(dimension: Optional[str] = None) -> dict[str, list[str]] | list[str]:
    """
    Return lexicon for a specific dimension or the full dict.
    dimension: "caste" | "gender" | "religion" | "race_ethnicity" | None
    """
    if dimension is None:
        return ENGLISH_LEXICONS
    if dimension not in ENGLISH_LEXICONS:
        raise ValueError(
            f"Unknown dimension {dimension!r}. "
            f"Choose from: {list(ENGLISH_LEXICONS.keys())}"
        )
    return ENGLISH_LEXICONS[dimension]


def get_all_terms_set() -> frozenset[str]:
    """Return frozen set of all English identity terms (lower-cased)."""
    return ENGLISH_ALL_TERMS


def count_identity_terms(text: str) -> dict[str, int]:
    """
    Count identity-term hits per dimension in `text`.
    Returns {dimension: count, "total": total_count}.
    """
    text_lower = text.lower()
    # Import here to avoid circular import at module-load time
    import re
    counts: dict[str, int] = {}
    total = 0
    for dim, terms in ENGLISH_LEXICONS.items():
        c = 0
        for term in terms:
            # Whole-word match (handles multi-word terms correctly)
            pattern = r"\b" + re.escape(term.lower()) + r"\b"
            c += len(re.findall(pattern, text_lower))
        counts[dim] = c
        total += c
    counts["total"] = total
    return counts


def get_term_positions(text: str) -> list[dict]:
    """
    Find all identity-term occurrences with position info.
    Returns list of {term, dimension, start, end, match}.
    Useful for annotation visualisation and Study 3 scoring.
    """
    import re
    text_lower = text.lower()
    hits = []
    for dim, terms in ENGLISH_LEXICONS.items():
        for term in terms:
            pattern = r"\b" + re.escape(term.lower()) + r"\b"
            for m in re.finditer(pattern, text_lower):
                hits.append({
                    "term":      term,
                    "dimension": dim,
                    "start":     m.start(),
                    "end":       m.end(),
                    "match":     text[m.start():m.end()],
                })
    hits.sort(key=lambda h: h["start"])
    return hits


def compute_gss(response_text: str) -> float:
    """
    Gendered Salience Score (GSS) for English text.
    GSS = identity_term_hits / total_token_count  (bounded [0, 1]).
    """
    if not response_text:
        return 0.0
    import re
    tokens = re.findall(r"\b\w+\b", response_text.lower())
    if not tokens:
        return 0.0
    hit_count = sum(
        1 for t in tokens if t in ENGLISH_ALL_TERMS
    )
    return hit_count / len(tokens)

