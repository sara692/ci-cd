from arabic_legal_rag.generation.citations import cited_numbers, sources_for

HITS = [
    {"article_number": 147, "citation": "Egyptian Civil Code, Article 147"},
    {"article_number": 89, "citation": "Egyptian Civil Code, Article 89"},
]


def test_cited_numbers_in_order_without_duplicates():
    assert cited_numbers("العقد شريعة المتعاقدين [147]. ويتم بالتراضي [89] [147]") == [147, 89]


def test_arabic_indic_digits_in_brackets():
    assert cited_numbers("حسب [١٤٧]") == [147]


def test_invented_article_is_dropped():
    assert sources_for("نص [999] و [147]", HITS) == ["Egyptian Civil Code, Article 147"]


def test_no_citation_no_sources():
    assert sources_for("إجابة بلا مصادر", HITS) == []
