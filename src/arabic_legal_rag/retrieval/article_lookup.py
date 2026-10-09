import re

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_SING = r"(?:article|art\.?|المادة|الماده|مادة|ماده)"
_PLUR = r"(?:articles|arts\.?|المواد|مواد)"
_NUM = r"\d{1,4}"
_SEP = r"\s*(?:,|،|and|&|و)\s*"
_MARK = r"(?:no\.?|number|رقم|#)?\s*"
_NOT_LATIN = (
    r"(?<![A-Za-z])"  # blocks "part 147" / "start 15", still allows Arabic prefixes (بالمادة)
)

_BEFORE = re.compile(rf"{_NOT_LATIN}{_SING}\s*{_MARK}({_NUM})", re.IGNORECASE)  # article 147
_PLURAL = re.compile(
    rf"{_NOT_LATIN}{_PLUR}\s*{_MARK}({_NUM}(?:{_SEP}{_NUM})*)", re.IGNORECASE
)  # articles 147 and 148
_AFTER = re.compile(
    rf"(?<![\d.])({_NUM})\s*(?:{_SING}|{_PLUR})(?!\w)", re.IGNORECASE
)  # 147 article


def extract_article_numbers(question: str) -> list[int]:
    q = question.translate(_DIGITS)
    found: list[tuple[int, int]] = []
    for m in _BEFORE.finditer(q):
        found.append((m.start(), int(m.group(1))))
    for m in _PLURAL.finditer(q):
        found += [(m.start(), int(n)) for n in re.findall(_NUM, m.group(1))]
    for m in _AFTER.finditer(q):
        found.append((m.start(), int(m.group(1))))
    out: list[int] = []
    for _, n in sorted(found):
        if n not in out:
            out.append(n)
    return out
