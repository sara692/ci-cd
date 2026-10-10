import os
from collections import deque

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("rag_requests_total", "ask requests", ["status"])
LATENCY = Histogram("rag_request_seconds", "ask latency", buckets=(0.5, 1, 2, 4, 8, 16, 32))
TOKENS = Counter("rag_tokens_total", "LLM tokens", ["type"])
COST = Counter("rag_cost_usd_total", "estimated LLM cost in USD")
FAITHFULNESS_ROLLING = Gauge(
    "rag_faithfulness_rolling", "mean faithfulness of the last 20 scored answers"
)

PRICE_IN = float(os.environ.get("LLM_PRICE_IN_PER_M", "0"))
PRICE_OUT = float(os.environ.get("LLM_PRICE_OUT_PER_M", "0"))
_recent: deque[float] = deque(maxlen=20)


def request_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * PRICE_IN + completion_tokens * PRICE_OUT) / 1_000_000


def record_request(
    status: str, seconds: float, prompt_tokens: int = 0, completion_tokens: int = 0
) -> None:
    REQUESTS.labels(status).inc()
    LATENCY.observe(seconds)
    if prompt_tokens or completion_tokens:
        TOKENS.labels("prompt").inc(prompt_tokens)
        TOKENS.labels("completion").inc(completion_tokens)
        COST.inc(request_cost(prompt_tokens, completion_tokens))


def update_rolling_mean(value: float) -> float:
    _recent.append(value)
    return sum(_recent) / len(_recent)
