"""
src/retriever.py — Hybrid retrieval for RAG Engineering Assistant.

Retrieval pipeline:
    1. Dense: embed query → cosine similarity in ChromaDB → top-k chunks
    2. BM25:  tokenize query → BM25 score over all docs → top-k chunks
    3. Merge: Reciprocal Rank Fusion (RRF) to combine both ranked lists
    4. Rerank (optional): cross-encoder reranker pushes most-relevant chunk to top

Why hybrid?
    Engineering documents contain domain acronyms (ASME B16.5, NPSH, COP, ASHRAE 90.1)
    that pure semantic search can underrank if they don't appear in training data
    with enough frequency. BM25 handles exact-match acronyms perfectly; dense
    handles paraphrased queries. Combining them with RRF is additive.

Usage:
    from src.retriever import Retriever
    r = Retriever()
    chunks = r.retrieve("minimum insulation for 4-inch steam pipe", top_k=4)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_PATH = Path(os.getenv("CHROMA_PERSIST_PATH", "./chroma_db"))
COLLECTION_NAME = "engineering_docs"
DEFAULT_TOP_K = 4
RRF_K = 60  # standard RRF constant — larger K reduces the impact of rank differences


# ---------------------------------------------------------------------------
# ChromaDB + embedding setup (cached at module level for reuse in API)
# ---------------------------------------------------------------------------

_collection = None
_embedding_fn = None
_bm25_index = None
_bm25_docs: list[dict] = []


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_PATH))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
        if provider == "local":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            _embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        else:
            from langchain_openai import OpenAIEmbeddings
            _embedding_fn = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embedding_fn


def _get_bm25_index():
    """Build a BM25 index over all documents in ChromaDB (lazy, cached)."""
    global _bm25_index, _bm25_docs
    if _bm25_index is not None:
        return _bm25_index, _bm25_docs

    from rank_bm25 import BM25Okapi

    collection = _get_collection()
    # Fetch all documents (needed to build BM25 corpus)
    result = collection.get(include=["documents", "metadatas"])
    _bm25_docs = [
        {"text": doc, "id": id_, "metadata": meta}
        for doc, id_, meta in zip(result["documents"], result["ids"], result["metadatas"])
    ]
    tokenized = [doc["text"].lower().split() for doc in _bm25_docs]
    _bm25_index = BM25Okapi(tokenized)
    return _bm25_index, _bm25_docs


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------

def dense_retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Embed query → cosine similarity search in ChromaDB.

    Returns a list of dicts with keys: text, source, page, chunk_id, score.
    Scores are cosine distances (lower = more similar in ChromaDB's cosine space).
    """
    collection = _get_collection()
    embedding_fn = _get_embedding_fn()
    query_embedding = embedding_fn.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", ""),
            "page": meta.get("page", 0),
            "chunk_id": f"{meta.get('source', '')}__p{meta.get('page', 0)}__c{meta.get('chunk_index', 0)}",
            "score": 1.0 - dist,  # convert distance to similarity
            "retrieval_method": "dense",
        })
    return chunks


# ---------------------------------------------------------------------------
# BM25 retrieval
# ---------------------------------------------------------------------------

def bm25_retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    BM25 keyword retrieval over the full ChromaDB document corpus.

    Returns a list of dicts with keys: text, source, page, chunk_id, score.
    """
    bm25, docs = _get_bm25_index()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices sorted by score descending
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]

    chunks = []
    for idx in top_indices:
        doc = docs[idx]
        chunks.append({
            "text": doc["text"],
            "source": doc["metadata"].get("source", ""),
            "page": doc["metadata"].get("page", 0),
            "chunk_id": doc["id"],
            "score": float(scores[idx]),
            "retrieval_method": "bm25",
        })
    return chunks


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = RRF_K,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = sum over ranked lists of 1 / (k + rank(d))
    where rank is 1-indexed. Documents not in a list are not penalized by
    RRF — they simply get no score contribution from that list.

    Args:
        ranked_lists: list of ranked result lists (each a list of chunk dicts)
        k: RRF smoothing constant (standard is 60)
        top_k: number of results to return

    Returns:
        Merged and re-ranked list of chunk dicts, with "rrf_score" added.
    """
    scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunk_lookup:
                chunk_lookup[cid] = chunk

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = []
    for chunk_id, rrf_score in merged:
        chunk = dict(chunk_lookup[chunk_id])
        chunk["rrf_score"] = rrf_score
        result.append(chunk)
    return result


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        # ms-marco-MiniLM-L-6-v2: fast, accurate, ~25MB download
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(query: str, chunks: list[dict]) -> list[dict]:
    """
    Re-rank chunks using a cross-encoder model.

    The cross-encoder sees the full (query, chunk) pair simultaneously —
    much more accurate than bi-encoder similarity but ~10× slower.
    Used after RRF to push the best chunk to rank 1.

    Returns chunks sorted by cross-encoder score descending.
    """
    reranker = _get_reranker()
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = reranker.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)


# ---------------------------------------------------------------------------
# Main retrieval entry point
# ---------------------------------------------------------------------------

class Retriever:
    """
    Hybrid retriever: dense + BM25 → RRF → optional cross-encoder rerank.

    Args:
        use_hybrid: combine dense + BM25 via RRF (default True)
        use_reranker: apply cross-encoder reranker after RRF (default True)
        top_k: number of chunks to return
    """

    def __init__(
        self,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Returns a list of chunk dicts, each with:
            text, source, page, chunk_id, score (or rrf_score), rerank_score
        """
        k = top_k or self.top_k

        if self.use_hybrid:
            dense = dense_retrieve(query, top_k=k * 2)  # over-fetch for fusion
            bm25 = bm25_retrieve(query, top_k=k * 2)
            chunks = reciprocal_rank_fusion([dense, bm25], top_k=k * 2)
        else:
            chunks = dense_retrieve(query, top_k=k * 2)

        if self.use_reranker and len(chunks) > 0:
            chunks = rerank(query, chunks)

        return chunks[:k]

    def retrieve_dense_only(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """Dense-only retrieval — used for ablation notebook 02."""
        return dense_retrieve(query, top_k=top_k or self.top_k)

    def retrieve_bm25_only(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """BM25-only retrieval — used for ablation notebook 02."""
        return bm25_retrieve(query, top_k=top_k or self.top_k)

    def retrieve_hybrid_no_rerank(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """Hybrid (RRF) without reranker — used for ablation notebook 04."""
        k = top_k or self.top_k
        dense = dense_retrieve(query, top_k=k * 2)
        bm25 = bm25_retrieve(query, top_k=k * 2)
        return reciprocal_rank_fusion([dense, bm25], top_k=k)
