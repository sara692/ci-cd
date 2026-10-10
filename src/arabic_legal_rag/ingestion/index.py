import hashlib
import json

import chromadb

from ..config import ROOT, load_params
from .chunker import build_chunks
from .embedder import embed, token_report


def build_index(params=None, manifest_path=None):
    p = params or load_params()
    corpus = ROOT / p["corpus_path"]
    records = json.loads(corpus.read_text(encoding="utf-8"))
    manifest_path = manifest_path or ROOT / "reports" / "index_manifest.json"

    chunks = build_chunks(records, p["chunk_text"])
    assert len(chunks) == len(records), "chunking must be 1:1 with articles"
    texts = [c["text"] for c in chunks]

    report = token_report([p["passage_prefix"] + t for t in texts], p["embedding_model"])
    vectors = embed(texts, p["embedding_model"], p["passage_prefix"], p["batch_size"])

    client = chromadb.PersistentClient(path=str(ROOT / p["index_dir"]))
    try:
        client.delete_collection(p["collection"])  # rebuild from scratch: reproducible
    except Exception:
        pass
    col = client.create_collection(p["collection"], metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), 500):
        part = chunks[i : i + 500]
        col.add(
            ids=[c["id"] for c in part],
            embeddings=vectors[i : i + 500],
            documents=[c["text"] for c in part],
            metadatas=[c["metadata"] for c in part],
        )

    manifest = {
        "n_chunks": col.count(),
        "embedding_model": p["embedding_model"],
        "chunk_text": p["chunk_text"],
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        **report,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    print(build_index())
