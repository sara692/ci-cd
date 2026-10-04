# tests/test_retrieval_smoke.py
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not Path("data/index").exists(), reason="index not built")


@pytest.mark.slow
def test_direct_lookup_returns_the_article():
    from arabic_legal_rag.rag import retrieve

    assert retrieve("ما نص المادة 147؟", 3)[0]["article_number"] == 147
