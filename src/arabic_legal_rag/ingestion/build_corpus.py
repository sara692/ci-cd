import json
import re
import sys

import pdfplumber

from .hierarchy import LEVELS, heading_level, heading_title, update_hierarchy
from .normalize_ar import clean_ar_cell

ART_EN = re.compile(r"^Article\s+(\d+)", re.I)
ART_AR = re.compile(
    r"^\u0645\u0627\u062f\u0629\s*\(?\s*(\d+)\s*\)?"
)  # مادة ( 89 ), after normalization
REPEAL_EN = re.compile(r"repeal|abolish|cancel", re.I)
REPEAL_AR = re.compile(r"ملغ|الغي")


def clean_en(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def build(pdf_path: str) -> list[dict]:
    records, current = [], None
    h = dict.fromkeys(LEVELS, "")

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    if not row or len(row) < 2:
                        continue
                    en_raw, ar_raw = row[0] or "", row[1] or ""
                    if not (en_raw.strip() or ar_raw.strip()):
                        continue
                    ar = clean_ar_cell(ar_raw)

                    m = ART_EN.match(en_raw.strip())
                    if m:  # new article row
                        if current:
                            records.append(current)
                        n = int(m.group(1))
                        am = ART_AR.match(ar)
                        current = {
                            "article_number": n,
                            **h,
                            "text_ar": ART_AR.sub("", ar, count=1).strip() if am else ar,
                            "text_en": clean_en(ART_EN.sub("", en_raw.strip(), count=1)),
                            "is_repealed": False,
                            "source_page": page.page_number,
                            "citation": f"Egyptian Civil Code, Article {n}",
                            "_ar_number": int(am.group(1)) if am else None,
                        }
                        continue

                    level = heading_level(en_raw)
                    if level:  # heading row
                        update_hierarchy(h, level, heading_title(en_raw, level))
                    elif current:  # continuation across a page break
                        current["text_ar"] = f'{current["text_ar"]} {ar}'.strip()
                        current["text_en"] = f'{current["text_en"]} {clean_en(en_raw)}'.strip()
        if current:
            records.append(current)

    for r in records:
        short = len(r["text_en"]) < 300
        r["is_repealed"] = short and bool(
            REPEAL_EN.search(r["text_en"]) or REPEAL_AR.search(r["text_ar"])
        )
        if r.pop("_ar_number") not in (None, r["article_number"]):
            print(f"WARNING: AR/EN number mismatch at article {r['article_number']}")
    return records


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    data = build(src)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(data)} articles -> {dst}")
