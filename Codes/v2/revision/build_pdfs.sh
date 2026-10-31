#!/usr/bin/env bash
# revision/build_pdfs.sh - compile the TCSS submission PDF with MiKTeX (pdflatex + bibtex).
#   bash revision/build_pdfs.sh
set -u
BIN="/c/Users/Debz/AppData/Local/Programs/MiKTeX/miktex/bin/x64"
cd "$(dirname "$0")/../../../Submission" || exit 1
for f in TCSS_Attentional_Harm; do
  "$BIN/pdflatex.exe" -interaction=nonstopmode "$f.tex" >/dev/null
  if grep -q '\\citation' "$f.aux" 2>/dev/null; then "$BIN/bibtex.exe" "$f" >/dev/null; fi
  "$BIN/pdflatex.exe" -interaction=nonstopmode "$f.tex" >/dev/null
  "$BIN/pdflatex.exe" -interaction=nonstopmode "$f.tex" >/dev/null
  pages=$(grep -a -o "Output written on .*(\([0-9]*\) pages" "$f.log" | sed 's/.*(\([0-9]*\) pages/\1/')
  errs=$(grep -c "^!" "$f.log")
  undef=$(grep -c "undefined" "$f.log")
  echo "$f: pages=$pages errors=$errs undefined=$undef"
done
