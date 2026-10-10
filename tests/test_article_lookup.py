import pytest

from arabic_legal_rag.retrieval.article_lookup import extract_article_numbers as f


@pytest.mark.parametrize(
    "q, want",
    [
        ("what is article 147", [147]),
        ("what is 147 article", [147]),
        ("article no. 147", [147]),
        ("المادة 147", [147]),
        ("147 مادة", [147]),
        ("ما نص المادة ١٤٧", [147]),
        ("بالمادة 147", [147]),
        ("explain articles 147 and 148", [147, 148]),
        ("what happens after 15 years", []),
        ("part 147", []),
    ],
)
def test_extract_article_numbers(q, want):
    assert f(q) == want
