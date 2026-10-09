FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False \
    RAG_ROOT=/app \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 1) dependencies only (cached until pyproject.toml / uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) the package (hatchling needs README.md)
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# 3) non-root user, then app data
RUN useradd --uid 1000 --create-home app
COPY params.yaml ./
COPY data/processed/civil_code.json data/processed/civil_code.json
COPY --chown=app:app data/index_bge data/index_bge
COPY reports/index_manifest.json reports/index_manifest.json

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "arabic_legal_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
