import json
from collections import Counter

data = json.load(open("data/processed/civil_code.json", encoding="utf-8"))
nums = [r["article_number"] for r in data]

print(f"records: {len(data)} | first: {min(nums)} | last: {max(nums)}")

missing = sorted(set(range(min(nums), max(nums) + 1)) - set(nums))
print(f"\nmissing numbers ({len(missing)}): {missing[:100]}")

dupes = [n for n, c in Counter(nums).items() if c > 1]
print(f"duplicate numbers: {dupes[:20]}")

out_of_order = [(a, b) for a, b in zip(nums, nums[1:], strict=False) if b <= a]
print(f"out-of-order pairs: {out_of_order[:20]}")

lens = sorted(((len(r["text_ar"]), r["article_number"]) for r in data), reverse=True)
print(f"\nlongest 8 (chars, article): {lens[:20]} at Arabic_text")
print(f"shortest 8 (chars, article): {lens[-20:]} at Arabic_text")

lens = sorted(((len(r["text_en"]), r["article_number"]) for r in data), reverse=True)
print(f"\nlongest 8 (chars, article): {lens[:20]} at English_text")
print(f"shortest 8 (chars, article): {lens[-20:]} at English_text")

print("\nrepealed flagged:", sorted(r["article_number"] for r in data if r["is_repealed"]))

print("\n--- articles 54-80: how do they look? ---")
for r in data:
    if 54 <= r["article_number"] <= 80:
        print(r["article_number"], r["is_repealed"], "|", r["text_en"][:70], "|", r["text_ar"][:40])

print("\n--- empty hierarchy by field ---")
for k in ("part", "book", "chapter", "section", "topic"):
    print(k, sum(1 for r in data if not r[k]))
