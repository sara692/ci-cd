"""BentoML service: async /ask and token-streaming /ask_stream (same logic as generation/answer.py)."""

import asyncio
import os
from collections.abc import AsyncGenerator

import bentoml
from bentoml.exceptions import InvalidArgument
from openai import AsyncOpenAI

from arabic_legal_rag import rag
from arabic_legal_rag.config import load_params
from arabic_legal_rag.generation.citations import sources_for
from arabic_legal_rag.generation.prompts import NOT_FOUND, REPEALED, build_messages, detect_lang


@bentoml.service(traffic={"timeout": 120, "concurrency": 32})
class RagService:
    def __init__(self) -> None:
        self.params = load_params()
        rag.documents_indexed()  # fail fast if the index is missing
        self.llm = AsyncOpenAI(
            base_url=os.environ["LLM_BASE_URL"], api_key=os.environ.get("LLM_API_KEY", "none")
        )

    @bentoml.api
    async def ask(self, question: str) -> dict:
        if not question.strip():
            raise InvalidArgument("question must not be empty")
        # rag.query is sync (embedding API + LLM call): run it in a worker thread so the event loop stays free
        return await asyncio.to_thread(rag.query, question)

    @bentoml.api
    async def ask_stream(self, question: str) -> AsyncGenerator[str, None]:
        if not question.strip():
            raise InvalidArgument("question must not be empty")
        p = self.params
        lang = detect_lang(question)
        hits = await asyncio.to_thread(rag.retrieve, question, p["top_k"])

        lookup = [h for h in hits if h["via"] == "lookup"]
        for h in lookup:
            if h["is_repealed"]:
                yield REPEALED[lang].format(n=h["article_number"])
                return
        context = lookup or [h for h in hits if h["score"] >= p["min_score"]]
        context = context[: p["max_context_articles"]]
        if not context:
            yield NOT_FOUND[lang]
            return

        stream = await self.llm.chat.completions.create(
            model=os.environ["LLM_MODEL"],
            messages=build_messages(question, context, lang),
            stream=True,
        )
        parts: list[str] = []
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                parts.append(token)
                yield token
        sources = sources_for("".join(parts), context)
        yield ("\n\nSources: " + "; ".join(sources)) if sources else ("\n\n" + NOT_FOUND[lang])
