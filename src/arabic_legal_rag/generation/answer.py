"""Answer pipeline: retrieved articles in, answer plus article citations out."""

from arabic_legal_rag.generation.citations import sources_for
from arabic_legal_rag.generation.prompts import (
    NOT_FOUND,
    REPEALED,
    build_messages,
    detect_lang,
)


def _result(answer: str, sources: list[str]) -> dict:
    return {"answer": answer, "sources": sources}


def _answer(question: str, retrieve, llm, params: dict) -> tuple[dict, list[dict]]:
    """Returns (result, context). context is what was sent to the LLM ([] if no LLM call)."""
    lang = detect_lang(question)
    hits = retrieve(question)

    lookup = [h for h in hits if h["via"] == "lookup"]
    for h in lookup:
        if h["is_repealed"]:
            return _result(REPEALED[lang].format(n=h["article_number"]), [h["citation"]]), []

    if lookup:
        context = lookup
    else:
        context = [h for h in hits if h["score"] >= params["min_score"]]
    context = context[: params["max_context_articles"]]

    if not context:
        return _result(NOT_FOUND[lang], []), []

    answer = llm(build_messages(question, context, lang))
    sources = sources_for(answer, context)
    if not sources:
        return _result(NOT_FOUND[lang], []), context
    return _result(answer, sources), context


def answer_question(question: str, retrieve, llm, params: dict) -> dict:
    """Used by rag.query() and the API."""
    return _answer(question, retrieve, llm, params)[0]


def answer_with_context(question: str, retrieve, llm, params: dict) -> tuple[dict, list[dict]]:
    """Used by evaluation: the result plus the articles the LLM actually saw."""
    return _answer(question, retrieve, llm, params)
