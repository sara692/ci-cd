import sys

import pdfplumber

PDF = "data/raw/egyptian_civil_code.pdf"
pages = [int(p) for p in sys.argv[1:]] or [1, 2, 3]

with pdfplumber.open(PDF) as pdf:
    print("total pages:", len(pdf.pages))
    for n in pages:
        page = pdf.pages[n - 1]
        tables = page.extract_tables()
        print(f"\n=== page {n}: {len(tables)} table(s) ===")
        for t in tables:
            print("rows:", len(t))
            for row in t[:4]:
                print([(c or "")[:70] for c in row])
