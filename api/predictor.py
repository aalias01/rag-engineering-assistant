"""
api/predictor.py — Retriever + Generator wiring for the FastAPI layer.

Loads the ChromaDB collection and initialises the Retriever/Generator once
at API startup (via FastAPI lifespan), then serves requests from cached state.

Graceful degraded mode: if ChromaDB isn't found (no documents ingested yet),
the API starts anyway and /query returns HTTP 503 with a clear message.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_PATH = Path(os.getenv("CHROMA_PERSIST_PATH", "./chroma_db"))

# Singletons — initialised in load_all()
_retriever = None
_generator = None
_collection_size = 0
_chroma_loaded = False


def load_all() -> None:
    """
    Initialise the retriever and generator. Called once at API startup
    via FastAPI lifespan. Sets _chroma_loaded = True on success.
    """
    global _retriever, _generator, _collection_size, _chroma_loaded

    try:
        from src.retriever import Retriever
        from src.generator import Generator

        # Verify ChromaDB exists and has documents
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_PATH))
        from src.ingestion import COLLECTION_NAME
        collection = client.get_collection(COLLECTION_NAME)
        _collection_size = collection.count()

        if _collection_size == 0:
            print("WARNING: ChromaDB collection is empty. Run `python -m src.ingestion` first.")
            return

        _retriever = Retriever(use_hybrid=True, use_reranker=True, top_k=4)
        _generator = Generator()
        _chroma_loaded = True
        print(f"RAG system loaded. Collection: {_collection_size} chunks.")

    except Exception as e:
        print(f"WARNING: Could not load RAG system — API running in degraded mode.\n{e}")


def is_ready() -> bool:
    return _chroma_loaded and _retriever is not None and _generator is not None


def query(
    q: str,
    top_k: int = 4,
    use_hybrid: bool = True,
    use_reranker: bool = True,
) -> dict:
    """
    Run retrieval + generation for a query.

    Returns the full result dict from Generator.generate(), plus retrieved chunks.
    Raises RuntimeError if the system is not loaded.
    """
    if not is_ready():
        raise RuntimeError(
            "RAG system not ready. Documents may not be ingested yet. "
            "Run `python -m src.ingestion` to ingest PDFs into ChromaDB."
        )

    from src.retriever import Retriever
    from src.generator import Generator

    # Allow per-request retriever config (for ablation experiments)
    retriever = _retriever
    if use_hybrid != True or use_reranker != True or top_k != 4:
        retriever = Retriever(use_hybrid=use_hybrid, use_reranker=use_reranker, top_k=top_k)

    chunks = retriever.retrieve(q, top_k=top_k)
    result = _generator.generate(q, chunks)
    result["chunks"] = chunks
    return result


async def stream_query(q: str, top_k: int = 4):
    """
    Async generator for streaming responses (SSE).

    Usage in FastAPI:
        from fastapi.responses import StreamingResponse
        async def event_stream():
            async for token in predictor.stream_query(query):
                yield f"data: {token}\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    """
    if not is_ready():
        yield "data: ERROR: RAG system not ready. Run ingestion first.\n\n"
        return

    chunks = _retriever.retrieve(q, top_k=top_k)
    async for token in _generator.stream(q, chunks):
        yield token


def get_status() -> dict:
    return {
        "chroma_loaded": _chroma_loaded,
        "collection_size": _collection_size,
        "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
    }
