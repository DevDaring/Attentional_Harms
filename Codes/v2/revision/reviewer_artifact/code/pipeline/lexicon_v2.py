"""
lexicon_v2.py - layered, versioned identity lexicon with span matching.

Implements the lexical baselines required by Future_Plan.md s4:
  * longest non-overlapping spans, multiword terms, a versioned alias map;
  * pronouns kept in their own layer;
  * audited ambiguous words resolved by a context rule, otherwise reported
    as 'ambiguous' and excluded from the primary density;
  * counts are SPANS (one span may contain several tokens).

Sources for the term lists: WinoBias (Zhao et al., 2018), HolisticBias
(Smith et al., 2022), Indian-BhED (Khandelwal et al., 2024), extended by hand.
These counts are baselines to be compared against human relevance labels,
not a measure of harm.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

LEXICON_VERSION = "v2.1-2026-10-09"

LAYERS = ("pronoun", "role", "group_label", "identity_context")

PRONOUNS = {
    "he": "gender", "him": "gender", "his": "gender", "himself": "gender",
    "she": "gender", "her": "gender", "hers": "gender", "herself": "gender",
    "they": "gender", "them": "gender", "their": "gender", "theirs": "gender",
    "themself": "gender", "themselves": "gender",
    "xe": "gender", "xem": "gender", "xyr": "gender", "ze": "gender", "zir": "gender",
}
GENDERED_PRONOUNS = {"he", "him", "his", "himself", "she", "her", "hers", "herself"}

# canonical term -> category ; plural=True generates a regular plural alias
_GROUP = {
    "gender": ["woman", "man", "girl", "boy", "lady", "gentleman", "female", "male",
               "transgender", "trans", "non-binary", "nonbinary", "genderqueer", "genderfluid",
               "agender", "intersex", "cisgender", "transman", "transwoman", "hijra", "kinnar",
               "lgbtq", "lgbtqia", "queer", "gay", "lesbian", "bisexual"],
    "caste": ["dalit", "brahmin", "brahman", "kshatriya", "vaishya", "shudra", "savarna", "avarna",
              "scheduled caste", "scheduled castes", "scheduled tribe", "scheduled tribes", "adivasi",
              "obc", "other backward class", "other backward classes", "upper caste", "lower caste",
              "dominant caste", "oppressed caste", "untouchable", "harijan", "bahujan", "jatav",
              "valmiki", "chamar", "bhangi", "musahar", "pariah", "adi dravidar", "rajput",
              "jat", "kayastha", "baniya", "maratha", "ezhava", "nadar", "thevar", "vokkaliga",
              "lingayat", "kamma", "kapu", "namasudra", "matua", "kurmi", "koeri"],
    "religion": ["hindu", "muslim", "christian", "sikh", "jain", "buddhist", "jewish", "jew",
                 "parsi", "zoroastrian", "atheist", "agnostic", "catholic", "protestant",
                 "evangelical", "sunni", "shia", "sufi", "ahmadiyya", "bahai", "pagan", "infidel",
                 "kafir", "heretic"],
    "race_ethnicity": ["african american", "african-american", "black american", "asian american",
                       "caucasian", "hispanic", "latino", "latina", "latinx", "native american",
                       "indigenous", "aboriginal", "pacific islander", "arab", "middle eastern",
                       "south asian", "east asian", "southeast asian", "mixed race", "mixed-race",
                       "biracial", "multiracial", "person of colour", "people of colour",
                       "person of color", "people of color", "roma", "romani", "tribal",
                       "northeasterner", "dravidian", "aryan"],
}
# Nationality / regional demonyms: counted as race_ethnicity group labels with
# subtype 'nationality' so analyses can exclude them.
_NATIONALITY = ["indian", "pakistani", "bangladeshi", "nepali", "sri lankan", "chinese", "japanese",
                "korean", "vietnamese", "filipino", "african", "nigerian", "kenyan", "ethiopian",
                "somali", "ghanaian", "mexican", "cuban", "puerto rican", "jamaican", "american",
                "british", "european", "bengali", "punjabi", "tamil", "marathi", "gujarati",
                "bihari", "kashmiri", "assamese", "malayali", "telugu", "kannadiga", "odia"]
_ROLE = {
    "gender": ["mother", "father", "mom", "mum", "dad", "wife", "husband", "son", "daughter",
               "sister", "brother", "aunt", "uncle", "niece", "nephew", "grandmother", "grandfather",
               "granddaughter", "grandson", "girlfriend", "boyfriend", "bride", "groom", "widow",
               "widower", "housewife", "businessman", "businesswoman", "chairman", "chairwoman",
               "spokesman", "spokeswoman", "policeman", "policewoman", "actress", "waitress",
               "king", "queen", "prince", "princess", "mr", "mrs", "ms", "sir", "madam",
               "maam", "ma'am", "shri", "smt", "shrimati", "sahib", "begum", "babu", "bhai", "didi"],
    "religion": ["priest", "pastor", "imam", "rabbi", "monk", "nun", "pandit", "pujari", "maulvi",
                 "maulana", "granthi", "pope", "bishop", "swami", "sadhu", "sannyasi",
                 "mullah", "qazi", "missionary"],
}
_CONTEXT = {
    "religion": ["religion", "religious", "faith", "temple", "mandir", "mosque", "masjid", "church",
                 "cathedral", "chapel", "gurdwara", "synagogue", "monastery", "dargah", "islam",
                 "islamic", "hinduism", "christianity", "sikhism", "jainism", "buddhism", "judaism",
                 "diwali", "deepavali", "holi", "eid", "eid al-fitr", "eid al-adha", "ramadan",
                 "ramzan", "christmas", "easter", "gurpurab", "guru nanak jayanti", "vaisakhi",
                 "baisakhi", "navratri", "durga puja", "ganesh chaturthi", "onam", "pongal",
                 "hanukkah", "passover", "muharram", "lent", "puja", "namaz", "prayer", "halal",
                 "kosher", "sabbath", "bible", "quran", "gita", "bhagavad gita", "guru granth sahib",
                 "torah", "hijab", "burqa", "turban", "bindi", "sindoor", "mangalsutra"],
    "caste": ["caste", "castes", "casteism", "casteist", "jati", "varna", "untouchability",
              "reservation quota", "sc/st"],
    "gender": ["gender", "gendered", "feminine", "masculine", "womanhood", "manhood", "sexism",
               "misogyny", "pronoun", "pronouns"],
    "race_ethnicity": ["race", "racial", "racism", "racist", "ethnicity", "ethnic", "skin colour",
                       "skin color", "complexion", "fair skin", "dark skin"],
}
# Ambiguous words: counted only when a context rule resolves them.
_AMBIGUOUS = {
    "black": ("race_ethnicity", "group_label"), "white": ("race_ethnicity", "group_label"),
    "brown": ("race_ethnicity", "group_label"), "asian": ("race_ethnicity", "group_label"),
    "dom": ("caste", "group_label"), "mala": ("caste", "group_label"), "koli": ("caste", "group_label"),
    "mahar": ("caste", "group_label"), "madiga": ("caste", "group_label"),
    "meghwal": ("caste", "group_label"), "dhobi": ("caste", "group_label"),
    "orthodox": ("religion", "group_label"), "sect": ("religion", "identity_context"),
    "race": ("race_ethnicity", "identity_context"), "st": ("caste", "identity_context"),
    "miss": ("gender", "role"), "guru": ("religion", "role"),
    "polish": ("race_ethnicity", "group_label"),
    "sc": ("caste", "identity_context"),
}
PERSON_NOUNS = {"people", "person", "persons", "man", "men", "woman", "women", "community",
                "communities", "family", "families", "student", "students", "neighbour", "neighbours",
                "neighbor", "neighbors", "voter", "voters", "child", "children", "boy", "boys", "girl",
                "girls", "worker", "workers", "employee", "employees", "folk", "folks", "individual",
                "individuals", "candidate", "candidates", "patient", "patients", "customer",
                "customers", "resident", "residents", "citizen", "citizens", "americans", "british",
                "colleague", "colleagues", "author", "authors", "artist", "artists", "leader",
                "leaders", "caste", "castes", "group", "groups", "household", "households", "youth",
                "jews", "christians", "community's", "applicants", "applicant", "member", "members",
                "teacher", "teachers", "doctor", "doctors", "writer", "writers", "athlete", "athletes",
                "owner", "owners", "lives", "culture", "heritage", "history", "identity",
                "entrepreneur", "entrepreneurs", "founder", "founders", "professional", "professionals",
                "nurse", "nurses", "engineer", "engineers", "lawyer", "lawyers", "journalist",
                "journalists", "activist", "activists", "player", "players", "tenant", "tenants",
                "staff", "colleagues'", "parent", "parents", "mother", "mothers", "father", "fathers",
                "graduate", "graduates", "scholar", "scholars", "volunteer", "volunteers", "migrant",
                "migrants", "immigrant", "immigrants", "traveller", "travellers", "traveler", "travelers",
                "musician", "musicians", "officer", "officers", "manager", "managers", "owned", "led"}
_RELIGION_FOLLOW = {"christian", "christians", "jew", "jews", "jewish", "church", "churches",
                    "hindu", "hindus", "muslim", "muslims", "judaism", "christianity"}
_CASTE_ACRONYM_CONTEXT = {"students", "student", "category", "categories", "reservation", "quota",
                          "candidates", "community", "communities", "certificate", "scholarship",
                          "obc", "and", "/", "family", "families", "background", "household"}

# ---- v2.1 extension (9 Oct 2026): general lists of Indian caste/community names and of
# ethnic, tribal and regional groups, compiled from general knowledge across states and
# regions (not from the study prompts). Because many community names are also surnames,
# every name in _COMMUNITY counts only in an identity context (see _community_context).
_COMMUNITY = [
    # north and central
    "ahir", "gujjar", "gurjar", "koeri", "kushwaha", "lodhi", "lodh", "saini", "kamboj", "khatri",
    "arora", "bania", "agarwal", "bhumihar", "thakur", "pasi", "kori", "khatik", "mallah", "nishad",
    "kahar", "teli", "nai", "lohar", "sonar", "kumhar", "gadaria", "bairwa", "meena", "regar", "raigar",
    "mali", "dhanuk", "paswan", "dusadh", "ravidassia", "ramdasia", "mazhabi", "ramgarhia", "labana",
    "yadav", "kurmi", "jat", "rajput", "kayastha", "baniya", "brahmin",
    # west
    "kunbi", "matang", "mang", "chambhar", "dhangar", "vanjari", "agri", "bhandari", "patidar", "patel",
    "thakor", "rabari", "bharwad", "charan", "vankar", "lohana", "kanbi", "maratha", "patil",
    # south
    "reddy", "kamma", "kapu", "velama", "vokkaliga", "lingayat", "kuruba", "idiga", "vanniyar",
    "thevar", "mukkulathor", "nadar", "gounder", "chettiar", "mudaliar", "pillai", "iyer", "iyengar",
    "paraiyar", "pallar", "arunthathiyar", "adi dravida", "nair", "ezhava", "thiyya", "pulaya", "paraya",
    "nambudiri", "bunt", "billava", "mogaveera", "devanga", "naidu",
    # east and north-east
    "mahishya", "namasudra", "rajbanshi", "bagdi", "bauri", "kaibartta", "sadgop", "matua",
    "khandayat", "karan", "gope", "kalita", "koch", "mahato", "mahto", "kurmi mahato", "bhuiyan",
    # categories written as words
    "ebc", "mbc", "general category",
]
_COMMUNITY_NEXT = {"family", "families", "community", "communities", "caste", "castes", "samaj",
                   "household", "households", "background", "clan", "colony", "people", "man", "woman",
                   "men", "women", "boy", "girl", "student", "students", "groom", "bride", "origin",
                   "neighbour", "neighbours", "elders"}
_COMMUNITY_PREV = {"a", "an", "am", "are", "is", "as", "being", "was", "were", "fellow"}
_ETHNIC = [
    # India
    "mizo", "naga", "khasi", "garo", "bodo", "santal", "santhal", "oraon", "munda", "gond", "bhil",
    "baiga", "toda", "kodava", "siddi", "lepcha", "bhutia", "sikkimese", "tibetan", "gorkha", "gurkha",
    "manipuri", "meitei", "kuki", "tripuri", "ahom", "anglo-indian", "chinese-indian", "sindhi",
    "marwari", "dogra", "ladakhi", "andamanese", "great andamanese", "jarawa", "northeast indian",
    "north-east indian", "northeastern", "north-eastern",
    # elsewhere
    "kurdish", "kurd", "sami", "inuit", "maori", "torres strait islander", "igbo", "yoruba", "hausa",
    "zulu", "amhara", "oromo", "berber", "amazigh", "peranakan", "malay", "sinhalese", "pashtun",
    "chicano", "chicana", "mestizo", "uyghur", "rohingya", "hmong", "navajo", "cherokee",
    "first nations", "black british", "arab american", "mexican-american", "indian-american",
    "british pakistani", "british bangladeshi",
]
_DEMONYMS = [
    "afghan", "albanian", "algerian", "argentine", "argentinian", "armenian", "australian", "austrian",
    "azerbaijani", "bahraini", "belarusian", "belgian", "bhutanese", "bolivian", "bosnian", "brazilian",
    "bulgarian", "burmese", "cambodian", "cameroonian", "canadian", "chilean", "colombian", "congolese",
    "croatian", "cypriot", "czech", "danish", "dominican", "dutch", "ecuadorian", "egyptian", "emirati",
    "eritrean", "estonian", "fijian", "finnish", "french", "georgian", "german", "greek", "guatemalan",
    "haitian", "honduran", "hungarian", "icelandic", "indonesian", "iranian", "iraqi", "irish",
    "israeli", "italian", "ivorian", "jordanian", "kazakh", "kuwaiti", "kyrgyz", "lao", "latvian",
    "lebanese", "liberian", "libyan", "lithuanian", "malagasy", "malawian", "malaysian", "maldivian",
    "malian", "maltese", "mauritian", "moldovan", "mongolian", "montenegrin", "moroccan", "mozambican",
    "namibian", "nicaraguan", "nigerien", "norwegian", "omani", "palestinian", "panamanian",
    "paraguayan", "peruvian", "portuguese", "qatari", "romanian", "russian", "rwandan",
    "salvadoran", "saudi", "scottish", "senegalese", "serbian", "sierra leonean", "singaporean",
    "slovak", "slovenian", "south african", "spanish", "sudanese", "swedish", "swiss", "syrian",
    "taiwanese", "tajik", "tanzanian", "thai", "tibetan", "togolese", "tunisian", "turkish", "turkmen",
    "ugandan", "ukrainian", "uruguayan", "uzbek", "venezuelan", "welsh", "yemeni", "zambian",
    "zimbabwean", "caribbean", "afro-caribbean", "jamaican", "trinidadian", "barbadian", "guyanese",
    "cree", "ojibwe", "mohawk", "lakota", "hopi", "sioux",
]
_EXTRA_ROLE = {"gender": ["landlady", "headmistress", "headmaster", "hostess", "stewardess",
                          "nani", "dadi", "mausi", "chachi", "bhabhi"]}
_EXTRA_CONTEXT = {"religion": ["hebrew school", "sunday school", "madrasa", "madrasah", "seminary",
                               "yeshiva", "bible study", "langar", "satsang"],
                  "race_ethnicity": ["northeast india", "north-east india"]}

_IRREGULAR_PLURALS = {"woman": "women", "man": "men", "gentleman": "gentlemen", "wife": "wives",
                      "businessman": "businessmen", "businesswoman": "businesswomen",
                      "chairman": "chairmen", "chairwoman": "chairwomen", "spokesman": "spokesmen",
                      "spokeswoman": "spokeswomen", "policeman": "policemen",
                      "policewoman": "policewomen", "transman": "transmen", "transwoman": "transwomen",
                      "person of colour": "people of colour", "person of color": "people of color",
                      "lady": "ladies", "jew": "jews", "pastor": "pastors", "child": "children"}
_NO_PLURAL = {"female", "male", "transgender", "trans", "non-binary", "nonbinary", "genderqueer",
              "genderfluid", "agender", "intersex", "cisgender", "lgbtq", "lgbtqia", "queer",
              "jewish", "hispanic", "indigenous", "multiracial", "biracial", "mixed race", "mixed-race",
              "middle eastern", "east asian", "south asian", "southeast asian", "obc", "mr", "mrs",
              "ms", "miss", "sir", "madam", "maam", "ma'am", "shri", "smt", "shrimati", "religious",
              "islamic", "racial", "ethnic", "casteist", "racist", "gendered", "feminine",
              "masculine", "sc/st", "sc", "st", "scheduled castes", "scheduled tribes",
              "other backward classes", "people of colour", "people of color"}

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’\-/][A-Za-z]+)*")


@dataclass(frozen=True)
class Entry:
    canonical: str
    category: str
    layer: str
    subtype: str = ""
    ambiguous: bool = False


def _tok(text: str) -> list[tuple[str, int, int]]:
    out = []
    for m in _TOKEN_RE.finditer(text):
        w = m.group(0).replace("’", "'")
        low = w.lower()
        if low.endswith("'s"):
            low = low[:-2]
        out.append((low, m.start(), m.end()))
    return out


def _plural(term: str) -> str:
    if term in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[term]
    head, _, last = term.rpartition(" ")
    if last.endswith(("s", "sh", "ch", "x")):
        last = last + "es"
    elif last.endswith("y") and len(last) > 2 and last[-2] not in "aeiou":
        last = last[:-1] + "ies"
    else:
        last = last + "s"
    return (head + " " + last).strip()


def _build() -> dict[tuple[str, ...], Entry]:
    table: dict[tuple[str, ...], Entry] = {}

    def add(term: str, entry: Entry, plural: bool = True) -> None:
        forms = {term}
        if plural and term not in _NO_PLURAL:
            forms.add(_plural(term))
        for f in forms:
            key = tuple(t for t, _, _ in _tok(f))
            if key and key not in table:
                table[key] = entry

    for cat, terms in _GROUP.items():
        for t in terms:
            add(t, Entry(t, cat, "group_label"))
    for t in _NATIONALITY:
        add(t, Entry(t, "race_ethnicity", "group_label", subtype="nationality"))
    for cat, terms in _ROLE.items():
        for t in terms:
            add(t, Entry(t.replace("'", ""), cat, "role"))
    for cat, terms in _CONTEXT.items():
        for t in terms:
            add(t, Entry(t, cat, "identity_context"), plural=False)
    for t, (cat, layer) in _AMBIGUOUS.items():
        add(t, Entry(t, cat, layer, ambiguous=True))
    for cat, terms in _EXTRA_ROLE.items():
        for t in terms:
            add(t, Entry(t, cat, "role"))
    for cat, terms in _EXTRA_CONTEXT.items():
        for t in terms:
            add(t, Entry(t, cat, "identity_context"), plural=False)
    for t in _ETHNIC:
        add(t, Entry(t, "race_ethnicity", "group_label", subtype="ethnic"))
    for t in _DEMONYMS:
        add(t, Entry(t, "race_ethnicity", "group_label", subtype="nationality"))
    for t in _COMMUNITY:
        add(t, Entry(t, "caste", "group_label", subtype="community", ambiguous=True))
    for p, cat in PRONOUNS.items():
        table[(p,)] = Entry(p, cat, "pronoun")
    return table


_TABLE = _build()
_MAX_LEN = max(len(k) for k in _TABLE)


def _resolve_ambiguous(entry: Entry, tokens: list[tuple[str, int, int]], i: int, j: int,
                       text: str) -> bool:
    """True when the context rule confirms the identity reading."""
    nxt = tokens[j][0] if j < len(tokens) else ""
    nxt2 = tokens[j + 1][0] if j + 1 < len(tokens) else ""
    c = entry.canonical
    if entry.subtype == "community":
        prev = tokens[i - 1][0] if i > 0 else ""
        return nxt in _COMMUNITY_NEXT or prev in _COMMUNITY_PREV
    if c in ("black", "white", "brown", "asian", "polish"):
        return nxt in PERSON_NOUNS or (nxt in ("and", "or") and nxt2 in PERSON_NOUNS)
    if c in ("dom", "mala", "koli", "mahar", "madiga", "meghwal", "dhobi"):
        prev = tokens[i - 1][0] if i > 0 else ""
        return (nxt in PERSON_NOUNS or nxt in ("caste", "castes", "community")
                or nxt in _COMMUNITY_NEXT or prev in _COMMUNITY_PREV)
    if c == "orthodox":
        return nxt in _RELIGION_FOLLOW
    if c == "sect":
        return True if any(t[0] in _RELIGION_FOLLOW for t in tokens[max(0, i - 4): j + 4]) else False
    if c == "race":
        window = {t[0] for t in tokens[max(0, i - 3): j + 3]}
        return bool(window & {"racial", "racism", "ethnicity", "ethnic", "people", "skin", "colour",
                              "color", "religion", "caste", "gender", "mixed", "discrimination"})
    if c == "miss":
        return j < len(tokens) and text[tokens[j][1]:tokens[j][1] + 1].isupper()
    if c == "guru":
        return bool({t[0] for t in tokens[max(0, i - 3): j + 3]} & {"nanak", "granth", "sikh", "gurdwara", "ashram", "spiritual"})
    if c in ("sc", "st"):
        surface = text[tokens[i][1]:tokens[j - 1][2]]
        window = {t[0] for t in tokens[max(0, i - 2): j + 2]}
        return surface.isupper() and bool(window & _CASTE_ACRONYM_CONTEXT)
    return False


def find_spans(text: str) -> list[dict]:
    """Greedy longest-match, left to right, non-overlapping."""
    if not text:
        return []
    tokens = _tok(text)
    spans, i = [], 0
    while i < len(tokens):
        match = None
        for n in range(min(_MAX_LEN, len(tokens) - i), 0, -1):
            key = tuple(t[0] for t in tokens[i:i + n])
            if key in _TABLE:
                match = (n, _TABLE[key])
                break
        if match is None:
            # Hyphenated compounds (British-Jamaican, Afro-Caribbean, Bengali-origin): match when
            # any part is a group label; the compound is counted once.
            word = tokens[i][0]
            if "-" in word:
                for part in word.split("-"):
                    e = _TABLE.get((part,))
                    if e and e.layer == "group_label" and not e.ambiguous:
                        match = (1, Entry(word, e.category, "group_label", subtype=e.subtype or "compound"))
                        break
            if match is None:
                i += 1
                continue
        n, entry = match
        j = i + n
        resolved = True
        if entry.ambiguous:
            resolved = _resolve_ambiguous(entry, tokens, i, j, text)
        start, end = tokens[i][1], tokens[j - 1][2]
        spans.append({"surface": text[start:end], "canonical": entry.canonical,
                      "category": entry.category, "layer": entry.layer, "subtype": entry.subtype,
                      "ambiguous": entry.ambiguous, "resolved": resolved, "start": start, "end": end})
        i = j
    return spans


def word_count(text: str) -> int:
    return len(_tok(text or ""))


def summarise(text: str) -> dict:
    """Span counts by layer and category, plus densities per 100 words."""
    spans = find_spans(text)
    n = word_count(text)
    counted = [s for s in spans if s["resolved"]]
    by_layer = {layer: sum(1 for s in counted if s["layer"] == layer) for layer in LAYERS}
    by_cat = {}
    for s in counted:
        if s["layer"] != "pronoun":
            by_cat[s["category"]] = by_cat.get(s["category"], 0) + 1
    non_pronoun = by_layer["role"] + by_layer["group_label"] + by_layer["identity_context"]
    groups = sorted({s["canonical"] for s in counted if s["layer"] == "group_label"})
    return {
        "n_words": n,
        "n_pronoun": by_layer["pronoun"],
        "n_gendered_pronoun": sum(1 for s in counted if s["canonical"] in GENDERED_PRONOUNS),
        "n_role": by_layer["role"],
        "n_group_label": by_layer["group_label"],
        "n_identity_context": by_layer["identity_context"],
        "n_ambiguous_unresolved": sum(1 for s in spans if not s["resolved"]),
        "n_nationality": sum(1 for s in counted if s["subtype"] == "nationality"),
        "density_non_pronoun": 100.0 * non_pronoun / n if n else 0.0,
        "density_group_label": 100.0 * by_layer["group_label"] / n if n else 0.0,
        "group_labels": groups,
        "categories_present": sorted(by_cat),
    }


def group_label_set(text: str, include_nationality: bool = True) -> set[str]:
    return {s["canonical"] for s in find_spans(text)
            if s["resolved"] and s["layer"] == "group_label"
            and (include_nationality or s["subtype"] != "nationality")}
