import pytest
from backend.app.services.chunking import chunk_text
from backend.app.services.ingestion import clean_text, table_to_markdown, detect_repeated_headers_footers

def test_clean_text():
    # Broken hyphenation test
    raw = "This is a demon-\nstration of hyphen-\n   ated text."
    cleaned = clean_text(raw)
    assert "demonstration" in cleaned
    assert "hyphenated" in cleaned

    # Extra spaces test
    spaces = "Too   many    spaces\n\n\n\nand newlines."
    cleaned_spaces = clean_text(spaces)
    assert "Too many spaces" in cleaned_spaces
    assert "\n\n\n" not in cleaned_spaces

def test_table_to_markdown():
    table = [
        ["Product", "Price", "Qty"],
        ["Apple", "$1.00", "10"],
        ["Banana", "$0.50", "20"]
    ]
    md = table_to_markdown(table)
    assert "| Product | Price | Qty |" in md
    assert "| --- | --- | --- |" in md
    assert "| Apple | $1.00 | 10 |" in md

def test_detect_repeated_headers_footers():
    pages = [
        "Confidential Company Report 2026\nBody content page 1\nPage 1 of 3",
        "Confidential Company Report 2026\nBody content page 2\nPage 2 of 3",
        "Confidential Company Report 2026\nBody content page 3\nPage 3 of 3"
    ]
    headers, footers = detect_repeated_headers_footers(pages)
    assert "Confidential Company Report 2026" in headers

def test_chunking_max_size_enforcement():
    # Large text test
    long_text = ("This is a comprehensive overview of advanced retrieval augmented generation. " * 50)
    chunks = chunk_text(long_text)
    assert len(chunks) > 1
    # Every chunk must be <= max chunk size
    from backend.app.config import settings
    for c in chunks:
        assert len(c) <= settings.MAX_CHUNK_SIZE
