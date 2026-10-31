"""revision/style_check.py <tex> - abstract length, long sentences, banned words, US spellings."""
import re
import sys

t = open(sys.argv[1], encoding="utf-8").read()
if "\\begin{abstract}" in t:
    a = t.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
    print("abstract words", len(a.split()))
start = "\\section{Introduction}" if "\\section{Introduction}" in t else "\\begin{document}"
body = t.split(start)[1].split("\\section*{Ethics statement}")[0]
body = re.sub(r"\\begin\{(table|figure|algorithm|sidewaystable)\*?\}.*?\\end\{\1\*?\}", "", body, flags=re.S)
body = re.sub(r"%[^\n]*", "", body)
plain = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", "", body)
sents = re.split(r"(?<=[.?!])\s+(?=[A-Z0-9])", plain)
words = [len(s.split()) for s in sents if len(s.split()) > 2]
print("sentences", len(words), "mean words %.1f" % (sum(words) / len(words)))
for s in sents:
    if len(s.split()) > 30:
        print("LONG", len(s.split()), "|", " ".join(s.split())[:170])
banned = ["leverage", "harness", "utiliz", "utilis", "showcase", "pave", "pioneer", "delve", "foster", "underscore",
          "unlock", "paradigm", "holistic", "seamless", "pivotal", "paramount", "landscape", "realm",
          "multifaceted", "intricate", "Furthermore", "Moreover", "Additionally", "Notably", "Importantly",
          "It is important", "It is worth", "To this end", "In this section", "This section presents",
          "In what follows", "robust", "comprehensive", "state-of-the-art", "significant", "novel",
          "always", "never", "guarantee", "the first", "unprecedented"]
for b in banned:
    n = len(re.findall(b, body))
    if n:
        print("CHECK word:", b, n)
for us in ["behavior", "modeling", "labeling", "analyze", "optimization", "summariz", "recogniz", "organiz",
           "characteriz", "neighbor", "color", "favor"]:
    n = len(re.findall(us, body))
    if n:
        print("US spelling:", us, n)
for m in re.finditer(r"\n\n(While|However)", body):
    print("paragraph starts with", m.group(1))

# label/ref and citation checks (main file plus generated tables)
import glob as _g
import os as _os
full = t + "".join(open(f, encoding="utf-8").read()
                   for f in _g.glob(_os.path.join(_os.path.dirname(_os.path.abspath(sys.argv[1])), "ipm_tables_v2", "*.tex")))
BS = chr(92)
labels = set(re.findall(BS * 2 + r"label\{((?:tab|fig):[^}]*)\}", full))
refs = set(re.findall(BS * 2 + r"ref\{([^}]*)\}", full))
print("unreferenced labels:", sorted(labels - refs))
print("refs without label:", sorted(r for r in refs if r.startswith(("tab", "fig")) and r not in labels))
print("raw cite:", len(re.findall(BS * 2 + r"cite\{", full)), " double-dollar:", full.count("$$"),
      " paren-math:", full.count(BS + "("))
