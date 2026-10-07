from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arabic_legal_rag.api.main import app


@pytest.mark.slow
@pytest.mark.skipif(not Path("data/index").exists(), reason="index not built")
def test_health_against_real_index():
    with TestClient(app) as c:  # runs the lifespan: checks the manifest, loads index + model
        body = c.get("/health").json()
    assert body["status"] == "healthy" and body["documents_indexed"] > 1000
