from arabic_legal_rag.generation.answer import answer_question
from arabic_legal_rag.generation.prompts import NOT_FOUND, detect_lang

P = {"min_score": 0.5, "max_context_articles": 5}


def hit(n, score=0.9, via="dense", repealed=False):
    return {
        "article_number": n,
        "citation": f"Egyptian Civil Code, Article {n}",
        "is_repealed": repealed,
        "score": score,
        "via": via,
        "text": f"text of {n}",
    }


def test_detect_lang():
    assert detect_lang("ما هي شروط العقد؟") == "ar"
    assert detect_lang("What is a contract?") == "en"


def test_answer_with_valid_citation():
    out = answer_question("ما هي شروط العقد؟", lambda q: [hit(89)], lambda m: "التراضي [89]", P)
    assert out["sources"] == ["Egyptian Civil Code, Article 89"]


def test_uncited_answer_is_rejected():
    out = answer_question("ما هي شروط العقد؟", lambda q: [hit(89)], lambda m: "إجابة بلا مصدر", P)
    assert out == {"answer": NOT_FOUND["ar"], "sources": []}


def test_low_scores_mean_not_found_and_llm_not_called():
    def boom(m):
        raise AssertionError("LLM must not be called")

    out = answer_question("What is the capital?", lambda q: [hit(5, score=0.2)], boom, P)
    assert out["sources"] == []


def test_repealed_article_skips_llm():
    def boom(m):
        raise AssertionError("LLM must not be called")

    out = answer_question("ما نص المادة 60؟", lambda q: [hit(60, 1.0, "lookup", True)], boom, P)
    assert "ملغاة" in out["answer"] and out["sources"] == ["Egyptian Civil Code, Article 60"]
