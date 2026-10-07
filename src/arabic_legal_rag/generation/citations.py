import re

from ..ingestion.normalize_ar import normalize_ar

# "[147]" - also tolerates "[المادة 147]" and "[Article 147]" if the model adds a word.
_CITE = re.compile(r"\[\s*(?:المادة|مادة|Article)?\s*(\d{1,4})\s*\]", re.I)


def cited_numbers(answer: str) -> list[int]:
    """Article numbers the answer cites, in order of first appearance."""
    seen: list[int] = []
    for m in _CITE.findall(normalize_ar(answer)):  # normalize: Arabic-Indic digits -> Western
        n = int(m)
        if n not in seen:
            seen.append(n)
    return seen


def sources_for(answer: str, hits: list[dict]) -> list[str]:
    """Citations built from retrieval metadata. A number that was never retrieved is dropped."""
    by_number = {h["article_number"]: h["citation"] for h in hits}
    return [by_number[n] for n in cited_numbers(answer) if n in by_number]
