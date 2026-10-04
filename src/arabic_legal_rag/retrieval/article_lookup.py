import re

from ..ingestion.normalize_ar import normalize_ar

# After normalize_ar() digits are Western, so one pattern covers "المادة ١٤٧" and "Article 147".
_REF = re.compile(r"(?:المادة|مادة|article|art\.?)\s*(?:رقم\s*)?\(?\s*(\d{1,4})\s*\)?", re.I)


def extract_article_numbers(question: str) -> list[int]:
    seen: list[int] = []
    for m in _REF.findall(normalize_ar(question)):
        n = int(m)
        if n not in seen:
            seen.append(n)
    return seen
