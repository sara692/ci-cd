import chromadb

from ..config import ROOT, load_params
from ..ingestion.embedder import embed
from ..ingestion.normalize_ar import normalize_ar
from .article_lookup import extract_article_numbers


def _hit(id_, doc, meta, score, via):
    return {
        "article_number": meta["article_number"],
        "citation": meta["citation"],
        "is_repealed": meta["is_repealed"],
        "score": score,
        "text": doc,
        "metadata": meta,
        "via": via,
    }


class Retriever:
    def __init__(self, params: dict | None = None):
        self.p = params or load_params()
        client = chromadb.PersistentClient(path=str(ROOT / self.p["index_dir"]))
        self.col = client.get_collection(self.p["collection"])

    def count(self) -> int:
        return self.col.count()

    def search(self, question: str, k: int | None = None, where: dict | None = None) -> list[dict]:
        k = k or self.p["top_k"]
        hits: list[dict] = []

        numbers = extract_article_numbers(question)  # exact article references first
        if numbers:
            res = self.col.get(
                ids=[f"art-{n}" for n in numbers], include=["documents", "metadatas"]
            )
            hits += [
                _hit(i, d, m, 1.0, "lookup")
                for i, d, m in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
            ]

        vec = embed([normalize_ar(question)], self.p["embedding_model"], self.p["query_prefix"])[0]
        res = self.col.query(
            query_embeddings=[vec],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        seen = {h["article_number"] for h in hits}
        for i, d, m, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
            strict=True,
        ):
            if m["article_number"] not in seen:
                hits.append(_hit(i, d, m, 1 - dist, "dense"))
        return hits[:k]
