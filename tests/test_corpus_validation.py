import re

from arabic_legal_rag.ingestion.normalize_ar import clean_ar_cell, normalize_ar


def to_visual(logical: str) -> str:
    """Simulate pdfplumber output: reverse the line, keep numbers readable."""
    return re.sub(r"\d+", lambda m: m.group()[::-1], logical[::-1])


def test_round_trip_multi_digit():
    logical = "مادة (147)"
    assert clean_ar_cell(to_visual(logical)) == logical


def test_noisy_brackets():
    assert normalize_ar("( 1 (") == "(1)"
    assert normalize_ar(")2(") == "(2)"


def test_indic_digits():
    assert normalize_ar("مادة ( ١٤٧ )") == "مادة (147)"
