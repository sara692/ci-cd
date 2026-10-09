import json
import logging
import os
import time
from contextlib import asynccontextmanager

import openai
from fastapi import FastAPI, HTTPException
from prometheus_client import make_asgi_app

from .. import rag
from ..config import ROOT, load_params
from ..observability import metrics
from ..observability.tracing import enabled as tracing_enabled
from ..observability.tracing import traced_query
from .schemas import AskRequest, AskResponse, HealthResponse

log = logging.getLogger("rag.api")


def check_index_matches_params() -> None:
    """Fail at startup if the index was built with other settings than params.yaml."""
    manifest = ROOT / "reports" / "index_manifest.json"
    if not manifest.exists():
        log.warning("no index manifest found; cannot verify the index")
        return
    built = json.loads(manifest.read_text(encoding="utf-8"))
    wanted = load_params()
    for key in ("embedding_model", "chunk_text"):
        if built.get(key) != wanted[key]:
            raise RuntimeError(
                f"index built with {key}={built.get(key)!r} but params.yaml says "
                f"{wanted[key]!r}: rebuild with `uv run python -m arabic_legal_rag.ingestion.index`"
            )


def llm_configured() -> bool:
    return bool(os.getenv("LLM_BASE_URL") and os.getenv("LLM_MODEL"))


def embeddings_configured() -> bool:
    return bool(os.getenv("EMBED_API_KEY"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    check_index_matches_params()
    rag.documents_indexed()  # opens the index; fails fast if it is missing
    if embeddings_configured():
        try:
            rag.retrieve("warm-up", 1)  # checks the embedding API key and route once
        except openai.OpenAIError:
            log.exception("embedding API warm-up failed; /ask will answer 503 until it works")
    else:
        log.warning("EMBED_API_KEY not set: /ask will answer 503")
    if not llm_configured():
        log.warning("LLM_BASE_URL / LLM_MODEL not set: /ask will answer 503")
    yield
    if tracing_enabled():  # send the last traces before the process ends
        from langfuse import get_client

        get_client().flush()


app = FastAPI(title="Arabic Legal RAG", version="0.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return {"status": "healthy", "documents_indexed": rag.documents_indexed()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> dict:
    start = time.perf_counter()
    if not (llm_configured() and embeddings_configured()):
        metrics.record_request("unavailable", time.perf_counter() - start)
        raise HTTPException(status_code=503, detail="LLM or embedding backend is not configured")
    try:
        result, _trace_id = traced_query(req.question, rag.query)
    except openai.OpenAIError as exc:
        metrics.record_request("unavailable", time.perf_counter() - start)
        log.exception("LLM or embedding call failed")
        raise HTTPException(status_code=503, detail="LLM or embedding backend unavailable") from exc
    except Exception:
        metrics.record_request("error", time.perf_counter() - start)
        raise
    metrics.record_request("ok", time.perf_counter() - start)
    return result
