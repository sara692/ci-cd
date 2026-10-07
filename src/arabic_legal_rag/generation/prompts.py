import re

_ARABIC = re.compile(r"[\u0600-\u06FF]")
_LETTER = re.compile(r"[^\W\d_]")

SYSTEM = {
    "ar": (
        "أنت مساعد قانوني يجيب عن الأسئلة المتعلقة بالقانون المدني المصري.\n"
        "القواعد:\n"
        "1. أجب فقط اعتمادا على المواد المرفقة في السياق. لا تستخدم أي معلومة من خارجها.\n"
        "2. بعد كل جملة تستند فيها إلى مادة، اكتب رقم المادة بين قوسين مربعين هكذا: [147].\n"
        "3. إذا لم تكفِ المواد المرفقة للإجابة، اكتب بالضبط: لا تتضمن المواد المتاحة إجابة عن هذا السؤال.\n"
        "4. إذا كانت المادة ملغاة فاذكر ذلك صراحة.\n"
        "5. أجب بالعربية وباختصار."
    ),
    "en": (
        "You are a legal assistant answering questions about the Egyptian Civil Code.\n"
        "Rules:\n"
        "1. Answer ONLY from the articles provided in the context. Use nothing from outside them.\n"
        "2. After every sentence that relies on an article, write its number in square brackets, "
        "like [147].\n"
        "3. If the provided articles do not contain the answer, write exactly: "
        "The available articles do not answer this question.\n"
        "4. If an article is repealed, say so explicitly.\n"
        "5. Answer in English and keep it concise."
    ),
}

NOT_FOUND = {
    "ar": "لا تتضمن المواد المتاحة إجابة عن هذا السؤال.",
    "en": "The available articles do not answer this question.",
}
REPEALED = {
    "ar": "المادة {n} ملغاة ولم تعد نافذة.",
    "en": "Article {n} has been repealed and is no longer in force.",
}


def detect_lang(text: str) -> str:
    letters = _LETTER.findall(text)
    if not letters:
        return "en"
    share = len(_ARABIC.findall(text)) / len(letters)
    return "ar" if share >= 0.5 else "en"


def build_context(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        tag = " (REPEALED)" if h["is_repealed"] else ""
        blocks.append(f'[{h["article_number"]}]{tag}\n{h["text"]}')
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[dict], lang: str) -> list[dict]:
    user = (
        f"السياق:\n{build_context(hits)}\n\nالسؤال: {question}"
        if lang == "ar"
        else (f"Context:\n{build_context(hits)}\n\nQuestion: {question}")
    )
    return [{"role": "system", "content": SYSTEM[lang]}, {"role": "user", "content": user}]
