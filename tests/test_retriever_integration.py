from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CHROMA_SQLITE = ROOT / "chroma_db" / "chroma.sqlite3"


@pytest.mark.skipif(
    not CHROMA_SQLITE.exists(),
    reason="committed Chroma store is missing",
)
def test_committed_store_bm25_retriever_returns_chunks(monkeypatch):
    monkeypatch.chdir(ROOT)

    from src.retriever import Retriever

    retriever = Retriever(use_hybrid=True, use_reranker=False, top_k=3)
    chunks = retriever.retrieve_bm25_only("relief valve pressure", top_k=3)

    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk["source"]
        assert isinstance(chunk["page"], int)
        assert chunk["text"]
