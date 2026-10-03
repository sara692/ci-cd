import re
import sys

import pdfplumber

PHRASE = sys.argv[1] if len(sys.argv) > 1 else "borne by the owner of the dominant"
pdf = pdfplumber.open("data/raw/egyptian_civil_code.pdf")


def squash(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").lower()


hits = [n for n, p in enumerate(pdf.pages, 1) if PHRASE.lower() in squash(p.extract_text())]
print("pages whose TEXT contains the phrase:", hits)

for n in hits:
    for pn in (n - 1, n, n + 1):
        page = pdf.pages[pn - 1]
        tables = page.extract_tables()
        print(
            f"\n=== page {pn}: {len(tables)} table(s), "
            f"{sum(1 for r in page.rects if r.get('fill'))} filled rects ==="
        )
        for ti, t in enumerate(tables):
            print(f" table {ti}: {len(t)} rows; cell counts: {sorted({len(r) for r in t})}")
            for ri, row in enumerate(t):
                first = (row[0] or "").strip().replace("\n", " | ")[:55]
                print(f"   r{ri}: {first!r}")
        lines = [ln for ln in (page.extract_text() or "").splitlines() if "rticle" in ln]
        print(" 'rticle' lines in raw text:", [repr(x) for x in lines])
