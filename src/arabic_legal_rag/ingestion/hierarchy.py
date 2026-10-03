import re

LEVELS = ["part", "book", "chapter", "section", "topic"]
PATTERNS = {
    "part": re.compile(r"^(FIRST|SECOND|THIRD|FOURTH)\s+PART", re.I),
    "book": re.compile(r"^BOOK\s+[IVXL]+", re.I),
    "chapter": re.compile(r"^CHAPTER\s+[IVXL]+", re.I),
    "section": re.compile(r"^SECTION\s+[IVXL\d]+", re.I),
}
NUMBERED = re.compile(r"^\d+\.\s+\S")


def heading_level(en_cell: str) -> str | None:
    first = en_cell.strip().splitlines()[0] if en_cell.strip() else ""
    for level, pat in PATTERNS.items():
        if pat.match(first):
            return level
    if NUMBERED.match(first):  # numbered heading is now a topic
        return "topic"
    if first.rstrip().endswith(":") and len(first) < 80:
        return "topic"
    return None


def heading_title(en_cell: str, level: str) -> str:
    """'BOOK I\\nOBLIGATIONS GENERALLY' -> 'Obligations Generally'.

    topic: '1. Elements of Contracts\\nConsent:' -> 'Elements of Contracts > Consent'
    """
    lines = [ln.strip() for ln in en_cell.strip().splitlines() if ln.strip()]
    if level == "topic":
        parts = [re.sub(r"^\d+\.\s*", "", ln).rstrip(":").strip() for ln in lines]
        return " > ".join(p for p in parts if p)
    return " ".join(lines[1:]).title() if len(lines) > 1 else lines[0].title()


def update_hierarchy(h: dict, level: str, title: str) -> None:
    h[level] = title
    for lower in LEVELS[LEVELS.index(level) + 1 :]:  # a new Book resets Chapter, Section, ...
        h[lower] = ""
