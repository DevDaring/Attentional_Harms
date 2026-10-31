"""Apply the independent numerical audit's corrections (24 Oct 2026)."""
from pathlib import Path

SUB = Path(__file__).resolve().parents[3] / "Submission"
REV = Path(__file__).resolve().parent


def patch(path: Path, pairs):
    s = path.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, (path.name, a[:70])
        s = s.replace(a, b)
    path.write_text(s, encoding="utf-8")


patch(SUB / "IPM_Cover_Letter.tex", [
    ("Two adapted revision methods lost required identity in 6 of 9 and 3 of 10 eligible pairs while lowering identity-term counts, a loss that term-count evaluation scores as improvement.",
     "Two adapted revision methods lost required identity in 6 of 9 and 3 of 10 eligible pairs; 12 of the 13 retention losses coincided with a fall in identity-term count, which term-count evaluation scores as improvement."),
    ("All limitations are stated,", "The main limitations are stated,"),
    ("Deviations from the internally frozen protocol are dated and reported.",
     "Deviations from the internally frozen protocol are reported, with dates where they could be recovered."),
])
patch(SUB / "IPM_Highlights.tex", [
    ("an LLM judge failed validation.", "an LLM judge ensemble failed validation."),
    ("Adapted revision methods deleted required identity while term counts fell.",
     "Two revision methods deleted required identity while term counts fell."),
])
print("all patches applied")
