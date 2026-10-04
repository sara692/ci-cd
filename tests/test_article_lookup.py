from arabic_legal_rag.retrieval.article_lookup import extract_article_numbers


def test_arabic_indic_digits():
    assert extract_article_numbers("ما نص المادة ١٤٧؟") == [147]


def test_english_and_multiple():
    assert extract_article_numbers("Compare Article 147 with article 148") == [147, 148]


def test_no_reference():
    assert extract_article_numbers("ما هي شروط العقد؟") == []
