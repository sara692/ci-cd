import json
from collections import defaultdict

from arabic_legal_rag.rag import retrieve

rows = [
    json.loads(x) for x in open("data/eval/retrieval_probe.jsonl", encoding="utf-8") if x.strip()
]
by_lang = defaultdict(lambda: {"n": 0, "hit1": 0, "hit5": 0, "rr": 0.0})
top1: dict[int, dict[str, int]] = defaultdict(dict)

for r in rows:
    nums = [h["article_number"] for h in retrieve(r["question"], 5)]
    s = by_lang[r["lang"]]
    s["n"] += 1
    s["hit1"] += nums[:1] == [r["gold"]]
    s["hit5"] += r["gold"] in nums
    s["rr"] += 1 / (nums.index(r["gold"]) + 1) if r["gold"] in nums else 0
    top1[r["gold"]][r["lang"]] = nums[0]

for lang, s in by_lang.items():
    print(
        f'{lang}: hit@1={s["hit1"]/s["n"]:.2f} hit@5={s["hit5"]/s["n"]:.2f} MRR={s["rr"]/s["n"]:.2f} (n={s["n"]})'
    )

pairs = [v for v in top1.values() if {"ar", "en"} <= v.keys()]
same = sum(v["ar"] == v["en"] for v in pairs)
print(f"bilingual agreement (AR and EN question -> same top-1 article): {same}/{len(pairs)}")
