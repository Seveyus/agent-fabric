from core.retrieval.chunking import chunk_text


def test_chunk_text():
    text = "A" * 2500
    chunks = chunk_text(text, max_chars=1000, overlap=100)
    assert len(chunks) >= 2
