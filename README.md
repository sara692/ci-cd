# Arabic Legal RAG — Egyptian Civil Code Q&A

Ask questions about the **Egyptian Civil Code** in Arabic or English and get an answer that **cites the articles it is based on**.

The system is a retrieval-augmented generation (RAG) pipeline over all 1,094 articles of the code, served through an API, a chat UI and a monitoring stack, with CI checking answer quality on every change.

> **Not legal advice.** This is a study and engineering project. Answers can be wrong or incomplete. Check the cited article before relying on it.

---

## Contents

1. [How it works](#how-it-works)
2. [Project layout](#project-layout)
3. [Quick start](#quick-start)
4. [Configuration](#configuration)
5. [Running the services (with screenshots)](#running-the-services)
6. [Evaluation results](#evaluation-results)
7. [CI/CD](#cicd)
8. [Design decisions](#design-decisions)
9. [Not done / limitations](#not-done--limitations)

---

## How it works

```
question ─► normalize ─► embed (API) ─► Chroma search ─► select articles ─► LLM (API) ─► answer + citations
                          bge-m3         data/index_bge    min_score,         Gemma 31B
                                                           article lookup
```

- **One article = one chunk.** Each of the 1,094 articles is indexed as a single document with a single normalized Arabic text field (`text_ar`).
- **Article-number questions** ("المادة 147", "article 147") skip vector search and fetch that article directly.
- **Retrieval:** embeddings come from an API (`baai/bge-m3`, 1024 dimensions) and are stored in a local Chroma collection. Hits are sorted by score and cut to the top *k*.
- **Generation:** an LLM (Gemma 31B, through an OpenAI-compatible API) answers *only* from the retrieved articles. If nothing relevant is found, or the article is repealed, the system says so instead of guessing.
- **Citations:** sources are checked against the retrieved articles before they are returned.

---

## Project layout

```
.
├── src/arabic_legal_rag/
│   ├── ingestion/        # corpus normalisation, embedding, index build
│   ├── retrieval/        # Chroma search, article-number lookup
│   ├── generation/       # prompts, LLM client, answer + citation checks
│   ├── api/              # FastAPI app (/ask, /health, /metrics)
│   ├── serving/          # BentoML service (async + streaming)
│   ├── observability/    # Langfuse tracing, Prometheus metrics, scoring
│   ├── monitoring/       # RAGAS trend runner, drift detection
│   ├── evaluation/       # RAGAS evaluation, experiments, CI gate
│   └── rag.py            # query() entry point
├── ui/streamlit_app.py   # chat UI
├── tests/                # unit tests + load test (tests/load/locustfile.py)
├── data/
│   ├── processed/civil_code.json   # the corpus (1,094 articles)
│   └── eval/                       # question sets for evaluation
├── infra/monitoring/     # Prometheus, Alertmanager, Grafana config
├── docker-compose.yml            # the API
├── docker-compose.monitoring.yml # Prometheus + Alertmanager + Grafana
├── docker-compose.canary.yml     # stable/canary behind nginx
├── params.yaml           # pipeline settings
├── .github/workflows/ci.yml
└── docs/images/          # screenshots used in this README
```

---

## Quick start

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker (for the container and the monitoring stack), and API keys for an embeddings provider and an LLM provider.

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>

cp .env.example .env            # then edit .env (see Configuration)
uv sync --all-extras            # install the project and every optional extra

# build the vector index from the corpus (calls the embeddings API once)
uv run --no-sync python -m arabic_legal_rag.ingestion.index

# start the API
uv run --no-sync uvicorn arabic_legal_rag.api.main:app --port 8000
```

Open <http://localhost:8000/docs> and try `POST /ask`.

**With Docker instead:**

```bash
docker compose up --build
curl -s localhost:8000/health
```

---

## Configuration

Settings live in two places:

- `params.yaml` — pipeline settings (embedding model, `top_k`, `min_score`, chunk text, index folder).
- `.env` — secrets and endpoints. **Never commit `.env`.** Values must be plain `KEY=value` (no quotes, no `export`, no spaces around `=`), because Docker's `--env-file` keeps quotes literally.

| Variable | Purpose |
|---|---|
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | Generator (OpenAI-compatible API) |
| `JUDGE_BASE_URL`, `JUDGE_MODEL`, `JUDGE_API_KEY` | Judge model used for RAGAS scoring |
| `EMBED_BASE_URL`, `EMBED_API_KEY` | Embeddings API |
| `LLM_PRICE_IN_PER_M`, `LLM_PRICE_OUT_PER_M` | Price per million tokens, for the cost metric |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | Tracing (optional; tracing is off when unset) |
| `MLFLOW_TRACKING_URI` | Experiment tracking |

---

## Running the services

| Service | Port | Start |
|---|---|---|
| FastAPI | 8000 | `uv run --no-sync uvicorn arabic_legal_rag.api.main:app --port 8000` |
| Streamlit UI | 8501 | `uv run --no-sync streamlit run ui/streamlit_app.py` |
| BentoML | 3001 | `uv run --no-sync bentoml serve arabic_legal_rag.serving.service:RagService --port 3001` |
| Locust | 8089 | `uv run --no-sync locust -f tests/load/locustfile.py --host http://localhost:3001` |
| MLflow | 5000 | `uv run --no-sync mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts` |
| Langfuse | 3000 | `docker compose -p langfuse -f ~/langfuse/docker-compose.yml up -d` |
| Prometheus | 9095 | `docker compose -p monitoring -f docker-compose.monitoring.yml up -d` |
| Grafana | 3031 | (same compose file) |
| Alertmanager | 9093 | (same compose file) |

### FastAPI

`POST /ask` returns `{"answer": ..., "sources": [...]}`. `GET /health` reports the number of indexed articles. `GET /metrics` exposes Prometheus metrics. Invalid questions get `422`; an unreachable or unconfigured LLM or embeddings backend gets `503`.

> 📷 **Screenshot placeholder** — replace `docs/images/fastapi_docs.png` with the Swagger page (`/docs`) showing a successful `/ask` response.

![FastAPI Swagger page](docs/images/fastapi_docs.png)

### Streamlit chat UI

A chat page that calls the API, shows the answer with its sources and the response time, and renders Arabic right-to-left. It can talk to FastAPI (`/ask`) or to BentoML (`/ask_stream`, token by token).

> 📷 **Screenshot placeholder** — replace `docs/images/streamlit_ui.png` with the UI answering one Arabic and one English question.

![Streamlit UI](docs/images/streamlit_ui.png)

### BentoML (async + streaming)

The same pipeline as an async service. `/ask_stream` sends tokens as they are produced.

```bash
curl -N -X POST localhost:3001/ask_stream -H "Content-Type: application/json" -d @/tmp/q_ar.json
```

> 📷 **Screenshot placeholder** — replace `docs/images/bentoml_ui.png` with the service page at <http://localhost:3001> or a terminal showing the streamed output.

![BentoML service](docs/images/bentoml_ui.png)

### Locust load test

50 simulated users calling `/ask` and `/ask_stream` against the BentoML service. Requests go through the embeddings and LLM providers, so their rate limits affect the result.

```bash
uv run --no-sync locust -f tests/load/locustfile.py --host http://localhost:3001 \
    -u 50 -r 5 -t 3m --headless --html reports/locust_50users.html --csv reports/locust
```

| Users | Requests/s | p50 (s) | p95 (s) | Failures |
|---|---|---|---|---|
| 50 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

> 📷 **Screenshot placeholder** — replace `docs/images/locust_ui.png` with the Locust statistics and charts tabs from the 50-user run.

![Locust results](docs/images/locust_ui.png)

### Prometheus

The API exposes `rag_requests_total`, `rag_request_seconds`, `rag_tokens_total`, `rag_cost_usd_total` and `rag_faithfulness_rolling`. Prometheus scrapes `/metrics` every 15 seconds.

> 📷 **Screenshot placeholder** — replace `docs/images/prometheus_targets.png` with **Status → Targets** at <http://localhost:9095/targets> showing the `rag` job **UP**.

![Prometheus targets](docs/images/prometheus_targets.png)

### Grafana

Dashboard panels: request rate, p95 latency, tokens per second, cost so far, rolling faithfulness.

> 📷 **Screenshot placeholder** — replace `docs/images/grafana_dashboard.png` with the dashboard at <http://localhost:3031>.

![Grafana dashboard](docs/images/grafana_dashboard.png)

**Alert:** `rag_faithfulness_rolling < 0.80` for 5 minutes fires `LowFaithfulness`, and Alertmanager sends a webhook notification.

> 📷 **Screenshot placeholder** — replace `docs/images/alert_fired.png` with the firing alert (Prometheus → Alerts) and the received webhook message.

![Alert fired](docs/images/alert_fired.png)

### Langfuse traces

Every `/ask` is recorded as a trace (retrieval, generation, latency). Answered questions get a faithfulness score attached in the background.

> 📷 **Screenshot placeholder** — replace `docs/images/langfuse_trace.png` with one trace showing its steps and the faithfulness score.

![Langfuse trace](docs/images/langfuse_trace.png)

_Tracing used: **self-hosted Langfuse** / **Langfuse Cloud** (delete one)._

### MLflow experiments

Each configuration (embedding model × chunk text) is one run; the best one is registered with the alias `production`.

> 📷 **Screenshot placeholder** — replace `docs/images/mlflow_comparison.png` with the run comparison chart.

![MLflow comparison](docs/images/mlflow_comparison.png)

### Canary rollout

`docker-compose.canary.yml` runs a stable and a canary copy behind nginx and splits traffic by percentage.

| Stage | Traffic to canary | Hold | Promote if | Roll back if |
|---|---|---|---|---|
| 1 | 5 % | 1 hour / 200 requests | all gates green | any gate red |
| 2 | 25 % | 2 hours | green | red |
| 3 | 50 % | 4 hours | green | red |
| 4 | 100 % | — | — | — |

Gates, compared with stable over the same window: faithfulness below 0.75 or more than 0.05 below stable; p95 latency above 1.3× stable; 5xx rate above 1 %; "not found" rate moving by more than 10 points. Rollback means setting the canary weight to 0 % and reloading nginx.

---

## Evaluation results

Quality is measured with [RAGAS](https://docs.ragas.io/) on a hand-written question set (answerable, off-topic and repealed-article questions, in Arabic and English), using a judge model.

| Metric | Value |
|---|---|
| Faithfulness | _TBD_ |
| Answer relevancy | _TBD_ |
| Context precision | _TBD_ |
| Context recall | _TBD_ |
| Answer rate | _TBD_ |
| Gold article cited | _TBD_ |
| p50 latency (s) | _TBD_ |

**Configuration comparison** (embedding model × chunk text):

| Run | Embedding model | Chunk text | Faithfulness | hit@1 | hit@5 |
|---|---|---|---|---|---|
| _TBD_ | | | | | |

**Judge model:** _TBD_ (note if it is the same model as the generator, since a model grading itself scores itself generously).

**Drift check (optional):** _TBD_ / not done.

---

## CI/CD

GitHub Actions runs on every pull request and on pushes to `main`:

```
lint ─► test ─► evaluate ─► docker (main only)
ruff    pytest   build the index,   build and push the image
black   -m "not  run the RAGAS gate
        slow"    (faithfulness ≥ 0.75)
```

- API keys are stored as repository **secrets** (`LLM_API_KEY`, `JUDGE_API_KEY`, `EMBED_API_KEY`); endpoints and model names as repository **variables**.
- The `evaluate` job fails the build if faithfulness on the CI question set drops below the threshold.
- Pull requests from forks do not receive secrets, so `evaluate` is skipped for them.

Run the checks locally:

```bash
uv run --no-sync ruff check src tests scripts
uv run --no-sync black --check src tests scripts
uv run --no-sync pytest -m "not slow" -q
```

---

## Design decisions

- **One chunk per article.** Legal citations are by article, so retrieval and citation use the same unit.
- **Embeddings and LLM through APIs** (OpenRouter for `baai/bge-m3`; an OpenAI-compatible endpoint for Gemma 31B). No GPU is needed, and the Docker image stays small.
- **Refuse instead of guess.** Low-similarity questions, off-topic questions and repealed articles get an explicit "not found" or "repealed" answer.
- **Tracing and scoring never break the API.** Without Langfuse keys, tracing is off; scoring runs in the background and swallows its own errors.
- **Secrets only in `.env` / GitHub secrets**, never in git or in the image.

---

## Not done / limitations

- **Self-hosted LLM serving (vLLM) and 4-bit quantization (AWQ):** not done. No GPU was available, and the generator is used through an API.
- **Re-ranker distillation:** not done / only if retrieval metrics showed the right article retrieved but ranked low. _(update)_
- **Drift detection:** _(update: done with cosine / MMD / classifier AUC, or not done)_
- **Token and cost metrics** stay at zero unless the LLM client records the response's token usage. _(update)_
- Answers depend on the generator and judge models; scores from a judge that is also the generator are optimistic.

---

## License and data

_Add your license here._ The corpus is the text of the Egyptian Civil Code; state its source and check that your use of it is permitted.
