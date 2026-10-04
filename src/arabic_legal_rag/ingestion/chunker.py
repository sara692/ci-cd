HIERARCHY = ("part", "book", "chapter", "section", "topic")


def build_text(r: dict, mode: str) -> str:
    ar, en = r["text_ar"].strip(), r["text_en"].strip()
    if mode == "ar_only":
        return ar or en  # English-only articles (e.g. 1022) fall back to English
    body = "\n".join(p for p in (ar, en) if p)
    if mode == "ar_en":
        return body
    if mode == "header_ar_en":
        crumbs = " > ".join(r[k] for k in HIERARCHY if r.get(k))
        head = r["citation"] + (f" | {crumbs}" if crumbs else "")
        return f"{head}\n{body}"
    raise ValueError(f"unknown chunk_text mode: {mode}")


def build_chunks(records: list[dict], mode: str) -> list[dict]:
    chunks = []
    for r in records:
        chunks.append(
            {
                "id": f'art-{r["article_number"]}',
                "text": build_text(r, mode),
                "metadata": {
                    "article_number": r["article_number"],
                    "citation": r["citation"],
                    "is_repealed": r["is_repealed"],
                    "has_arabic": bool(r["text_ar"].strip()),
                    "source_page": r["source_page"],
                    **{k: r.get(k) or "" for k in HIERARCHY},  # Chroma rejects None values
                },
            }
        )
    return chunks
