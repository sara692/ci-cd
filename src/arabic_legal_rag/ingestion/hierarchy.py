import re

LEVELS = ["part", "book", "chapter", "section", "subsection", "topic"]
PATTERNS = {
    "part": re.compile(r"^(FIRST|SECOND|THIRD|FOURTH)\s+PART", re.I),
    "book": re.compile(r"^BOOK\s+[IVXL]+", re.I),
    "chapter": re.compile(r"^CHAPTER\s+[IVXL]+", re.I),
    "section": re.compile(r"^SECTION\s+[IVXL\d]+", re.I),
    "subsection": re.compile(r"^\d+\.\s+\S"),
}


def heading_level(en_cell: str) -> str | None:
    first = en_cell.strip().splitlines()[0] if en_cell.strip() else ""
    for level, pat in PATTERNS.items():
        if pat.match(first):
            return level
    if first.rstrip().endswith(":") and len(first) < 80:
        return "topic"
    return None


def heading_title(en_cell: str, level: str) -> str:
    """'BOOK I\\nOBLIGATIONS GENERALLY' -> 'Obligations Generally'; topic -> 'Consent'."""
    lines = [ln.strip() for ln in en_cell.strip().splitlines() if ln.strip()]
    if level == "topic":
        return lines[0].rstrip(":").strip()
    if level == "subsection":
        return re.sub(r"^\d+\.\s*", "", " ".join(lines))
    return " ".join(lines[1:]).title() if len(lines) > 1 else lines[0].title()


def update_hierarchy(h: dict, level: str, title: str) -> None:
    h[level] = title
    for lower in LEVELS[LEVELS.index(level) + 1 :]:  # a new Book resets Chapter, Section, ...
        h[lower] = ""
