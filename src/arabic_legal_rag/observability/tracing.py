"""Langfuse tracing for /ask. A no-op when the LANGFUSE_* keys are not set."""

import os


def enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def traced_ask(question, *, retrieve, answer_with_context, llm, params):
    """Return (result, context, trace_id). trace_id is None when tracing is off."""
    if not enabled():
        hits = retrieve(question, params["top_k"])
        result, context = answer_with_context(question, lambda _q: hits, llm, params)
        return result, context, None

    from langfuse import get_client  # imported late: the package is in the optional `monitor` extra

    lf = get_client()
    with lf.start_as_current_observation(
        as_type="span", name="ask", input={"question": question}
    ) as root:
        with lf.start_as_current_observation(as_type="span", name="retrieve") as s:
            hits = retrieve(question, params["top_k"])
            s.update(output=[(h["article_number"], round(h["score"], 3), h["via"]) for h in hits])
        with lf.start_as_current_observation(
            as_type="generation", name="generate", model=os.environ.get("LLM_MODEL", "")
        ) as g:
            result, context = answer_with_context(question, lambda _q: hits, llm, params)
            g.update(output=result["answer"])
        root.update(output=result)
        trace_id = lf.get_current_trace_id()
    return result, context, trace_id


def traced_query(question, fn):
    """Run fn(question) inside one Langfuse span. Returns (result, trace_id)."""
    if not enabled():
        return fn(question), None
    from langfuse import get_client

    lf = get_client()
    with lf.start_as_current_observation(
        as_type="span", name="ask", input={"question": question}
    ) as root:
        result = fn(question)
        root.update(output=result)
        trace_id = lf.get_current_trace_id()
    return result, trace_id
