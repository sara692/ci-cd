from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness

from arabic_legal_rag.config import load_params
from arabic_legal_rag.evaluation.ragas_eval import judge_models
from arabic_legal_rag.observability.metrics import FAITHFULNESS_ROLLING, update_rolling_mean
from arabic_legal_rag.observability.tracing import enabled

_faith = None


def _metric():
    global _faith
    if _faith is None:  # build lazily: creating the judge client at import time slows startup
        j_llm, _ = judge_models(load_params())
        _faith = Faithfulness(llm=j_llm)
    return _faith


async def score_trace(trace_id, question, result, context) -> None:
    if not result.get(
        "citations"
    ):  # a refusal ("not found") has no claims to check; use your own field name
        return
    try:
        sample = SingleTurnSample(
            user_input=question,
            response=result["answer"],
            retrieved_contexts=[h["text"] for h in context],
        )
        value = await _metric().single_turn_ascore(sample)
    except Exception:
        return  # scoring must never break or slow the API
    FAITHFULNESS_ROLLING.set(update_rolling_mean(value))
    if enabled() and trace_id:
        from langfuse import get_client

        get_client().create_score(
            trace_id=trace_id, name="faithfulness", value=value, data_type="NUMERIC"
        )
