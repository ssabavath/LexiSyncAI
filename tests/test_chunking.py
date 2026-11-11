from src.pdf_ingest import split_into_chunks


def test_text_is_preserved():
    text = "Article 1\nThis is the first legal provision."

    chunks = split_into_chunks(text)

    assert chunks
    assert "Article 1" in chunks[0]
    assert "This is the first legal provision." in chunks[0]


def test_empty_text_returns_empty_list():
    chunks = split_into_chunks("")

    assert chunks == []