from functools import lru_cache

from .config import load_params
from .generation.answer import answer_question
from .generation.llm_client import make_llm
from .ingestion.index import build_index
from .retrieval.retriever import Retriever


@lru_cache(maxsize=1)
def _retriever() -> Retriever:
    return Retriever()


def ingest() -> dict:
    """JSON corpus -> chunk by article -> embed -> vector store."""
    _retriever.cache_clear()
    return build_index()


def retrieve(question: str, k: int | None = None, where: dict | None = None) -> list[dict]:
    return _retriever().search(question, k, where)


def documents_indexed() -> int:
    return _retriever().count()


@lru_cache(maxsize=1)
def _llm():
    return make_llm(load_params())


def query(question: str, llm=None) -> dict:
    """question -> {"answer": str, "sources": list[str]} (sources are article citations)."""
    return answer_question(question, retrieve, llm or _llm(), load_params())


print(retrieve("what is 147 article", 5))
