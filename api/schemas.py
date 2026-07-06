"""
api/schemas.py — Pydantic request/response models for RAG Engineering Assistant API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=1000, description="Natural-language engineering question")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of chunks to retrieve")
    use_hybrid: bool = Field(default=True, description="Use hybrid dense+BM25 retrieval (recommended)")
    use_reranker: bool = Field(default=True, description="Apply cross-encoder reranker")
    stream: bool = Field(default=False, description="Stream response tokens via SSE (use GET /query/stream for SSE)")

    model_config = {"json_schema_extra": {
        "example": {
            "query": "What is the difference between a relief valve and a safety valve?",
            "top_k": 4,
            "use_hybrid": True,
            "use_reranker": True,
        }
    }}


class SourceCitation(BaseModel):
    source: str = Field(..., description="PDF filename")
    page: int = Field(..., description="Page number in source document")


class ChunkPreview(BaseModel):
    text: str = Field(..., description="Chunk text (truncated to 300 chars for API response)")
    source: str
    page: int
    retrieval_method: Optional[str] = Field(None, description="dense / bm25 / hybrid")
    rrf_score: Optional[float] = Field(None, description="Reciprocal Rank Fusion score")
    rerank_score: Optional[float] = Field(None, description="Cross-encoder rerank score")


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceCitation]
    chunks_used: int
    chunks: list[ChunkPreview] = Field(description="Retrieved chunks (for transparency)")
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int
    model: str
    provider: str
    refused: bool = Field(
        ...,
        description=(
            "True when the generated answer contains the instructed refusal phrase. "
            "This is a heuristic because the model could phrase a decline differently; "
            "the eval set's 5 of 5 refusals used the instructed phrase."
        ),
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    chroma_loaded: bool
    collection_size: int
    version: str = "1.0.0"
    llm_provider: str
