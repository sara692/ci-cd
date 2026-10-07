FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ANONYMIZED_TELEMETRY=False \
    RAG_ROOT=/app \
    HF_HOME=/opt/hf \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 1. Dependencies (changes rarely, so this layer stays cached)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2. Embedding model (depends only on params.yaml, not on your code)
COPY params.yaml ./
RUN python -c "import yaml; from sentence_transformers import SentenceTransformer; SentenceTransformer(yaml.safe_load(open('params.yaml'))['embedding_model'])"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 3. Your code
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# 4. Data and index
RUN useradd --create-home --uid 1000 app
COPY data/processed/civil_code.json data/processed/civil_code.json
COPY --chown=app:app data/index data/index
COPY reports/index_manifest.json reports/index_manifest.json

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "arabic_legal_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
