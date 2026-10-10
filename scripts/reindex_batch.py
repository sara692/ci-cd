"""Incrementally add or update articles: data/new_docs/*.json -> existing Chroma index."""

import json

import chromadb

from arabic_legal_rag.config import ROOT, load_params
from arabic_legal_rag.ingestion.embedder import embed
from arabic_legal_rag.ingestion.index import (
    record_to_chunk,
)  # <- use the helper index.py already uses


def main() -> None:
    p = load_params()
    files = sorted((ROOT / "data" / "new_docs").glob("*.json"))
    records = [r for f in files for r in json.loads(f.read_text(encoding="utf-8"))]
    if not records:
        print("nothing to do")
        return

    col = chromadb.PersistentClient(path=str(ROOT / p["index_dir"])).get_collection(p["collection"])
    before = col.count()
    ids, docs, metas = [], [], []
    for r in records:
        id_, doc, meta = record_to_chunk(r, p)
        ids.append(id_), docs.append(doc), metas.append(meta)

    vecs = embed(docs, p["embedding_model"], p["passage_prefix"], p["batch_size"])
    col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vecs)
    print(f"upserted {len(ids)} articles: count {before} -> {col.count()}")

    done = ROOT / "data" / "new_docs" / "processed"
    done.mkdir(exist_ok=True)
    for f in files:
        f.rename(done / f.name)


if __name__ == "__main__":
    main()
