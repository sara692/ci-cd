from functools import lru_cache

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
