import os

from openai import OpenAI


def make_llm(params: dict):
    """Return a function messages -> answer text. Reads the endpoint from the environment."""
    client = OpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.getenv("LLM_API_KEY", "none"),
    )
    model = os.environ["LLM_MODEL"]

    def complete(messages: list[dict]) -> str:
        res = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=params["llm_temperature"],
            max_tokens=params["llm_max_tokens"],
        )
        return (res.choices[0].message.content or "").strip()

    return complete
