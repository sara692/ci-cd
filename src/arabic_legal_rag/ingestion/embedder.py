from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def get_model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


def embed(
    texts: list[str], model_name: str, prefix: str = "", batch_size: int = 32
) -> list[list[float]]:
    model = get_model(model_name)
    return model.encode(
        [prefix + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,  # cosine == dot product
        show_progress_bar=len(texts) > 64,
    ).tolist()


def token_report(texts: list[str], model_name: str) -> dict:
    """Silent truncation is the main risk with long articles, so measure it."""
    model = get_model(model_name)
    lens = [len(model.tokenizer(t)["input_ids"]) for t in texts]
    return {
        "max_seq_length": model.max_seq_length,
        "max_tokens": max(lens),
        "n_truncated": sum(n > model.max_seq_length for n in lens),
    }
