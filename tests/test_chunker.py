import json
from pathlib import Path

import pytest

from arabic_legal_rag.ingestion.chunker import build_chunks

CORPUS = Path("data/processed/civil_code.json")


@pytest.fixture(scope="module")
def records():
    return json.loads(CORPUS.read_text(encoding="utf-8"))


@pytest.mark.parametrize("mode", ["ar_only", "ar_en", "header_ar_en"])
def test_one_chunk_per_article(records, mode):
    chunks = build_chunks(records, mode)
    assert len(chunks) == len(records)
    assert len({c["id"] for c in chunks}) == len(chunks)
    assert all(c["text"].strip() for c in chunks)  # English-only articles still get text
    assert all(v is not None for c in chunks for v in c["metadata"].values())


def test_metadata_matches_article(records):
    for r, c in zip(records, build_chunks(records, "ar_en"), strict=True):
        assert c["metadata"]["article_number"] == r["article_number"]
        assert c["metadata"]["citation"] == r["citation"]
