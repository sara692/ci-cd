# Genuinely absent from the source PDF, each verified by eye.
DOCUMENTED_GAPS: dict[int, str] = {
    # 389: "Not present in the source PDF.",     # fill from MISSING RANGES
}
EMPTY_AR_OK: dict[int, str] = {
    1022: "English only in the source PDF.",
}
EXPECTED_REPEALED = set(range(54, 81))
MAX_CHARS = 2500
MIN_CHARS = 15
