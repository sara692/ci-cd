import json

from arabic_legal_rag.config import load_params
from arabic_legal_rag.ingestion.chunker import build_chunks
from arabic_legal_rag.ingestion.embedder import get_model

p = load_params()
model = get_model(p["embedding_model"])
records = json.load(open(p["corpus_path"], encoding="utf-8"))
for mode in ("ar_en", "ar_only"):
    chunks = build_chunks(records, mode)
    over = [
        (
            len(model.tokenizer(p["passage_prefix"] + c["text"])["input_ids"]),
            c["metadata"]["article_number"],
        )
        for c in chunks
    ]
    over = sorted((n, a) for n, a in over if n > model.max_seq_length)[::-1]
    print(mode, "->", len(over), "over", model.max_seq_length, ":", over)
