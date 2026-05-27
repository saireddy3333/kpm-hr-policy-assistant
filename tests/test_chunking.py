from __future__ import annotations

from rag import config
from rag.chunker import load_chunks, save_chunks
from rag.document_loader import save_clean_text


def test_chunks_have_required_metadata_and_reasonable_lengths():
    save_clean_text()
    save_chunks()
    chunks = load_chunks()
    assert chunks
    required = {
        "chunk_id",
        "section",
        "subsection",
        "title",
        "tags",
        "source",
        "industry",
        "jurisdiction",
        "version",
        "crossrefs",
        "text",
        "word_count",
        "naturally_short",
    }
    for chunk in chunks:
        assert required.issubset(chunk)
        assert chunk["source"] == config.SOURCE_NAME
        assert chunk["industry"] == "Automotive Manufacturing"
        assert chunk["jurisdiction"] == "Kentucky, USA"
        assert chunk["version"] == "v1.0"
        assert chunk["chunk_id"].startswith("KPM-HR-")
        assert isinstance(chunk["tags"], list)

    acceptable = [
        chunk
        for chunk in chunks
        if config.CHUNK_MIN_WORDS <= chunk["word_count"] <= config.CHUNK_MAX_WORDS
        or chunk["naturally_short"]
    ]
    assert len(acceptable) / len(chunks) >= 0.95
    assert all(chunk["word_count"] <= config.CHUNK_MAX_WORDS for chunk in chunks)


def test_processed_clean_text_is_created_from_hr_policy_only():
    save_clean_text()
    text = config.CLEAN_POLICY_PATH.read_text(encoding="utf-8")
    assert "Kentucky Precision Manufacturing" in text
    assert "5.6 Family & Medical Leave" in text
    assert "Conversational User Experience Guide" not in text
    assert "MSAI 631" not in text
