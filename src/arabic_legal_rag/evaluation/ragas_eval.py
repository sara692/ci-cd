"""RAGAS + pipeline metrics for one config. Judge is separate from the generator."""

import json
import math
import os
import statistics
import time
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from arabic_legal_rag.config import ROOT, load_params
from arabic_legal_rag.generation.answer import answer_with_context
from arabic_legal_rag.generation.llm_client import make_llm
from arabic_legal_rag.retrieval.retriever import Retriever


def load_items(path):
    return [
        json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()
    ]


def _env(name, fallback):
    return os.environ.get(f"JUDGE_{name}") or os.environ[f"LLM_{fallback}"]


def judge_models(params):
    llm = ChatOpenAI(
        base_url=_env("BASE_URL", "BASE_URL"),
        api_key=_env("API_KEY", "API_KEY"),
        model=_env("MODEL", "MODEL"),
        temperature=0,
    )
    emb = OpenAIEmbeddings(
        base_url=os.environ.get("EMBED_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.environ["EMBED_API_KEY"],
        model=params["judge_embedding_model"],
        check_embedding_ctx_length=False,  # else it sends token IDs, which non-OpenAI models reject
        chunk_size=16,  # keep batches small for the free route
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)


def run_pipeline(items, params):
    retriever = Retriever(params)  # adapt to your Retriever signature

    def retrieve(q):
        return retriever.search(q, params["top_k"])

    llm = make_llm(params)
    rows = []
    for it in items:
        t0 = time.perf_counter()
        result, context = answer_with_context(it["question"], retrieve, llm, params)
        rows.append(
            {
                **it,
                "answer": result["answer"],
                "sources": result["sources"],
                "contexts": [
                    h.get("text_ar") or h.get("text", "") for h in context
                ],  # match your hit field
                "latency": time.perf_counter() - t0,
            }
        )
    return rows


def _nan_to_none(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def pipeline_metrics(rows):
    ans = [r for r in rows if r["kind"] == "answerable"]
    answered = [r for r in ans if r["sources"]]
    gold_hit = [
        r
        for r in answered
        if any(f"Article {g}" in " ".join(r["sources"]) for g in r["gold_articles"])
    ]
    off = [r for r in rows if r["kind"] in ("offtopic", "repealed")]
    return {
        "answer_rate": len(answered) / max(len(ans), 1),
        "gold_cited_rate": len(gold_hit) / max(len(answered), 1),
        "refusal_accuracy": (
            sum(1 for r in off if r["kind"] == "offtopic" and not r["sources"])
            + sum(1 for r in off if r["kind"] == "repealed")
        )
        / max(len(off), 1),
        "latency_p50": statistics.median(r["latency"] for r in rows),
    }


def ragas_metrics(rows, params):
    # score only answered, answerable items: refusals have no claims to check
    scored = [r for r in rows if r["kind"] == "answerable" and r["sources"]]
    if not scored:
        return {}
    ds = EvaluationDataset.from_list(
        [
            {
                "user_input": r["question"],
                "retrieved_contexts": r["contexts"],
                "response": r["answer"],
                "reference": r["ground_truth"],
            }
            for r in scored
        ]
    )
    j_llm, j_emb = judge_models(params)
    res = evaluate(
        ds,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=j_llm,
        embeddings=j_emb,
        run_config=RunConfig(
            timeout=300, max_retries=3, max_workers=int(os.environ.get("JUDGE_WORKERS", 1))
        ),
        raise_exceptions=False,
    )
    df = res.to_pandas()
    names = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "llm_context_precision_with_reference": "context_precision",
        "context_recall": "context_recall",
    }
    out = {}
    for col, name in names.items():
        if col in df:
            out[name] = _nan_to_none(float(df[col].mean(skipna=True)))
            out[f"{name}_nan"] = int(df[col].isna().sum())  # failed judge calls
    return out


def evaluate_config(params, items):
    rows = run_pipeline(items, params)
    return {**pipeline_metrics(rows), **ragas_metrics(rows, params)}, rows


def main():
    params = load_params()
    items = load_items(ROOT / "data/eval/qa_50_ragas.jsonl")
    metrics, _ = evaluate_config(params, items)
    out = ROOT / "reports/ragas_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
