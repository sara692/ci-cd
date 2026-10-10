import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from arabic_legal_rag import rag
from arabic_legal_rag.api import main
from arabic_legal_rag.api.main import app

client = TestClient(app)  # no `with`: the lifespan (index + model loading) is not started

CITATION = "Egyptian Civil Code, Article 147"


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "fake")
    # the embeddings key is not available in CI; the endpoint only checks that it is set
    monkeypatch.setattr(main, "embeddings_configured", lambda: True)
    # bypass Langfuse: call the pipeline directly
    monkeypatch.setattr(main, "traced_query", lambda q, fn: (fn(q), None))
    monkeypatch.setattr(rag, "documents_indexed", lambda: 1094)
    monkeypatch.setattr(rag, "query", lambda q: {"answer": f"{q} [147]", "sources": [CITATION]})


@pytest.mark.parametrize("body", [{"question": ""}, {"question": "   "}, {}, {"question": 5}])
def test_invalid_question_returns_422(body):
    assert client.post("/ask", json=body).status_code == 422


def test_health_shape():
    r = client.get("/health")
    assert r.json() == {"status": "healthy", "documents_indexed": 1094}


def test_llm_not_configured_returns_503(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL")
    assert client.post("/ask", json={"question": "x"}).status_code == 503


def test_llm_failure_returns_503(monkeypatch):
    def boom(q):
        raise openai.APIConnectionError(request=httpx.Request("POST", "http://llm.invalid"))

    monkeypatch.setattr(rag, "query", boom)
    assert client.post("/ask", json={"question": "x"}).status_code == 503


def test_ask_returns_answer_and_article_citations():
    r = client.post("/ask", json={"question": "ما نص المادة 147؟"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "147" in body["answer"]
    assert body["sources"] == [CITATION]
