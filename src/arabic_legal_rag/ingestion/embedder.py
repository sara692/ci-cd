import os
import time

import numpy as np
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

_RETRY = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
# The API has no local tokenizer, so the context limit comes from .env.
MAX_SEQ_LENGTH = int(os.environ.get("EMBED_MAX_TOKENS", "8192"))


def _client() -> OpenAI:
    key = os.environ.get("EMBED_API_KEY")
    if not key:
        raise RuntimeError("EMBED_API_KEY is not set (put it in .env)")
    return OpenAI(
        base_url=os.environ.get("EMBED_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=key,
        timeout=60,
        max_retries=0,  # we retry ourselves below
    )


def _embed_batch(client: OpenAI, model_name: str, batch: list[str]) -> list[list[float]]:
    for attempt in range(6):
        try:
            r = client.embeddings.create(model=model_name, input=batch, encoding_format="float")
            return [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
        except _RETRY:
            if attempt == 5:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def embed(
    texts: list[str], model_name: str, prefix: str = "", batch_size: int = 32
) -> list[list[float]]:
    client = _client()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = [(prefix + t) or " " for t in texts[i : i + batch_size]]  # API rejects ""
        vecs = _embed_batch(client, model_name, batch)
        if len(vecs) != len(batch):
            raise RuntimeError(f"embedding API returned {len(vecs)} vectors for {len(batch)} texts")
        out.extend(vecs)
    arr = np.asarray(out, dtype=np.float32)
    arr /= np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12, None)  # cosine == dot
    return arr.tolist()


def token_report(texts: list[str], model_name: str) -> dict:
    """Estimate only: no local tokenizer. Arabic is about 3 characters per token, so this
    is deliberately conservative. Keys match the old report."""
    lens = [len(t) // 3 + 1 for t in texts]
    return {
        "max_seq_length": MAX_SEQ_LENGTH,
        "max_tokens": max(lens),
        "n_truncated": sum(n > MAX_SEQ_LENGTH for n in lens),
    }
