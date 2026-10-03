from arabic_legal_rag.ingestion.hierarchy import apply_heading, new_hierarchy, parse_heading


def test_section_with_topic_in_one_cell():
    rows = ["SECTION I", "Laws and their Applications", "1. Laws and Rights"]
    assert parse_heading(rows) == [
        ("section", "Laws and their Applications"),
        ("topic", "Laws and Rights"),
    ]


def test_two_topic_lines_are_joined():
    assert parse_heading(["1. Elements of Contracts", "Consent:"]) == [
        ("topic", "Elements of Contracts > Consent")
    ]


def test_all_caps_titles():
    assert parse_heading(["FIRST PART", "OBLIGATIONS OR PERSONAL RIGHTS"]) == [
        ("part", "Obligations or Personal Rights")
    ]
    assert parse_heading(["BOOK I", "OBLIGATIONS GENERALLY"]) == [("book", "Obligations Generally")]


def test_continuation_text_is_not_a_heading():
    assert parse_heading(["common of a dominant tenement interrupts the", "others."]) is None


def test_strict_ignores_colon_only_line():
    assert parse_heading(["Consent:"], strict=True) is None
    assert parse_heading(["Consent:"]) == [("topic", "Consent")]


def test_new_chapter_resets_lower_levels():
    h = new_hierarchy()
    apply_heading(h, [("section", "Contracts"), ("topic", "Consent")])
    apply_heading(h, [("chapter", "Effects")])
    assert h["section"] == "" and h["topic"] == ""
