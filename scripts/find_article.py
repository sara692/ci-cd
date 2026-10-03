import json
import sys

import pdfplumber

N = sys.argv[1] if len(sys.argv) > 1 else "1022"
PHRASE = sys.argv[2] if len(sys.argv) > 2 else "absence of an agreement to the contrary, the cost"

# 1. where did the text end up in the JSON?
data = json.load(open("data/processed/civil_code.json", encoding="utf-8"))
print(f"in JSON as record {N}:", any(str(r["article_number"]) == N for r in data))
for r in data:
    if PHRASE.lower() in r["text_en"].lower():
        print("phrase found inside article", r["article_number"], "(page", r["source_page"], ")")

# 2. what does the PDF give us on the page(s) that mention it?
pdf = pdfplumber.open("data/raw/egyptian_civil_code.pdf")
for pn, page in enumerate(pdf.pages, 1):
    if N not in (page.extract_text() or ""):
        continue
    print(f"\n=== page {pn} mentions {N} ===")
    for ti, t in enumerate(page.extract_tables()):
        for ri, row in enumerate(t):
            cells = [(c or "") for c in row]
            if any(N in c for c in cells) or any(PHRASE[:25].lower() in c.lower() for c in cells):
                print(f"table {ti} row {ri}: {len(row)} cells")
                for ci, c in enumerate(cells):
                    print(f"  cell {ci}: {c[:60]!r}  first codes: {[hex(ord(x)) for x in c[:12]]}")
