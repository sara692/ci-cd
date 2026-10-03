import json
import random

data = json.load(open("data/processed/civil_code.json", encoding="utf-8"))
for r in random.sample(data, 20):
    print("=" * 70)
    print(
        r["article_number"], "|", r["book"], ">", r["chapter"], ">", r["section"], ">", r["topic"]
    )
    print("AR:", r["text_ar"][:200])
    print("EN:", r["text_en"][:200])
    print("repealed:", r["is_repealed"], "| page:", r["source_page"])
