import re
import unicodedata

_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_NUMBER = re.compile(r"\d+")
_BRACKET_NUM = re.compile(r"[()]\s*(\d+)\s*[()]")


def fix_visual_order(cell: str) -> str:
    """Convert pdfplumber's visual-order Arabic into logical order, line by line.

    Must run BEFORE NFKC: lam-alef ligature glyphs are single characters in the
    visual stream, and expanding them first (to 'لا') and then reversing turns
    them into 'ال'. That is why your record shows 'ةيملاسلاا'.
    """
    fixed = []
    for line in cell.splitlines():
        rev = line[::-1]
        rev = _NUMBER.sub(lambda m: m.group()[::-1], rev)  # keep numbers left-to-right
        fixed.append(rev)
    return "\n".join(fixed)


def normalize_ar(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # expands presentation forms
    text = _TASHKEEL.sub("", text).replace("\u0640", "")  # diacritics, tatweel
    text = re.sub("[إأآٱ]", "ا", text)  # alef variants
    text = text.replace("ى", "ي")  # alef maqsura -> ya
    text = text.translate(_DIGIT_MAP)  # Arabic-Indic -> Western digits
    text = _BRACKET_NUM.sub(r"(\1)", text)  # '( 1 (' / ')2(' -> '(1)' / '(2)'
    return re.sub(r"\s+", " ", text).strip()


def clean_ar_cell(cell: str) -> str:
    """Use this in the extractor: reorder first, then normalize."""
    return normalize_ar(fix_visual_order(cell))
