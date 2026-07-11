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


class ValidationReport(BaseModel):
    """Post-generation citation validation (v2, interpret path only).

    Deterministic checks that the answer's citations point at chunks that
    were actually retrieved and that its numbers appear in them.
    """

    status: str = Field(..., description="passed | flagged | refused | not_applicable | off")
    citations_found: int = 0
    citations_valid: int = 0
    invalid_citations: list[str] = []
    numbers_found: int = 0
    numbers_grounded: int = 0
    ungrounded_numbers: list[str] = []
    sentences_checked: int = 0
    sentences_supported: int = 0
    unsupported_sentences: list[str] = []
    support_coverage: float = 1.0
    notes: str = ""


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
    # ------------------------------------------------------------------
    # v2 routing fields (all optional — pre-v2 clients ignore them)
    # ------------------------------------------------------------------
    route: str = Field(
        default="synthesized",
        description=(
            "Which path produced the answer: 'factual_lookup' (typed facts DB, "
            "deterministic — the LLM never generated the value), 'synthesized' "
            "(retrieval + LLM with citation validation), or 'clarification' "
            "(query too underspecified to answer responsibly)."
        ),
    )
    intent: Optional[str] = Field(
        default=None, description="Classified intent: lookup | interpret | clarify"
    )
    intent_method: Optional[str] = Field(
        default=None,
        description="Classifier backend that made the call (zero_shot | local | rules | fallback_*)",
    )
    fact_id: Optional[str] = Field(
        default=None, description="Matched fact id (factual_lookup route only)"
    )
    fact_status: Optional[str] = Field(
        default=None,
        description="Curation status of the matched fact: draft | verified",
    )
    validation: Optional[ValidationReport] = Field(
        default=None, description="Citation validation report (synthesized route only)"
    )
    clarification: Optional[str] = Field(
        default=None, description="Clarification question (clarification route only)"
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    chroma_loaded: bool
    collection_size: int
    version: str = "2.0.0"
    llm_provider: str
    router_enabled: bool = False
    intent_classifier: Optional[str] = None
    facts_loaded: int = 0
