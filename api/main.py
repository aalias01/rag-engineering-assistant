"""
api/main.py — FastAPI backend for RAG Engineering Assistant.

Endpoints:
  GET  /              — landing page (health + system info)
  GET  /health        — structured health check (for Render)
  POST /query         — retrieve + generate; returns full JSON response
  GET  /query/stream  — SSE streaming response (query via ?q=... param)

Deployment: Render (free tier) via render.yaml Blueprint.
OPENAI_API_KEY must be set as a secret environment variable on Render.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api import predictor
from api.schemas import HealthResponse, QueryRequest, QueryResponse, SourceCitation, ChunkPreview


# ---------------------------------------------------------------------------
# Lifespan — load ChromaDB + models once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        predictor.load_all()
    except Exception as e:
        print(f"WARNING: Startup load failed — API running in degraded mode.\n{e}")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Engineering Assistant API",
    description=(
        "Query engineering technical documents (ASHRAE, NASA, OSHA, ASME) in natural language. "
        "Hybrid retrieval (dense + BM25 + RRF) with cross-encoder reranking. "
        "Grounded answers with source citations — no hallucination."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS origins
#
# Local dev origins are always allowed. The deployed frontend origin is read from
# the FRONTEND_ORIGIN env var (comma-separated for multiple). Set this in Render
# after the frontend is deployed. Set both origins (comma-separated):
#   FRONTEND_ORIGIN=https://rag.alvinalias.com,https://rag-engineering-assistant.vercel.app
# and redeploy. No production URL is hardcoded.
_DEV_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]
_frontend_origin_env = os.getenv("FRONTEND_ORIGIN", "").strip()
_extra_origins = [o.strip() for o in _frontend_origin_env.split(",") if o.strip()]
_ALLOWED_ORIGINS = _DEV_ORIGINS + _extra_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
def root():
    status = predictor.get_status()
    return {
        "project": "RAG Engineering Assistant",
        "description": "Query engineering standards in natural language",
        "status": "ready" if status["chroma_loaded"] else "degraded — run ingestion",
        "collection_chunks": status["collection_size"],
        "llm_provider": status["llm_provider"],
        "docs": "/docs",
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    status = predictor.get_status()
    return HealthResponse(
        status="healthy" if status["chroma_loaded"] else "degraded",
        chroma_loaded=status["chroma_loaded"],
        collection_size=status["collection_size"],
        llm_provider=status["llm_provider"],
    )


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_endpoint(request: QueryRequest):
    """
    Retrieve + generate a grounded answer for an engineering question.

    The response includes the answer, source citations, the retrieved chunks
    (for transparency), token counts, and cost estimate.
    """
    if not predictor.is_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG system not ready. Documents may not be ingested yet. "
                "Run `python -m src.ingestion` from the project root."
            ),
        )

    t0 = time.time()
    try:
        result = predictor.query(
            q=request.query,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            use_reranker=request.use_reranker,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    latency_ms = int((time.time() - t0) * 1000)

    # Build response
    sources = [SourceCitation(source=s["source"], page=s["page"]) for s in result["sources"]]
    chunk_previews = [
        ChunkPreview(
            text=c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"],
            source=c.get("source", ""),
            page=c.get("page", 0),
            retrieval_method=c.get("retrieval_method"),
            rrf_score=c.get("rrf_score"),
            rerank_score=c.get("rerank_score"),
        )
        for c in result.get("chunks", [])
    ]

    return QueryResponse(
        query=request.query,
        answer=result["answer"],
        sources=sources,
        chunks_used=result["chunks_used"],
        chunks=chunk_previews,
        prompt_tokens=result.get("prompt_tokens", 0),
        completion_tokens=result.get("completion_tokens", 0),
        cost_usd=result.get("cost_usd", 0.0),
        latency_ms=latency_ms,
        model=result.get("model", ""),
        provider=result.get("provider", ""),
    )


# ---------------------------------------------------------------------------
# GET /query/stream — SSE streaming endpoint
# ---------------------------------------------------------------------------

@app.get("/query/stream", tags=["RAG"])
async def query_stream(
    q: str = Query(..., min_length=5, description="Engineering question"),
    top_k: int = Query(default=4, ge=1, le=10),
):
    """
    Stream the answer token-by-token via Server-Sent Events (SSE).

    The frontend connects with EventSource API:
        const es = new EventSource(`/query/stream?q=${encodeURIComponent(query)}`);
        es.onmessage = (e) => appendToken(e.data);

    The final event is prefixed with `__METADATA__:` and contains JSON with
    sources, cost, and token counts.
    """
    if not predictor.is_ready():
        async def error_stream():
            yield "data: ERROR: RAG system not ready. Run ingestion first.\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def event_stream():
        async for token in predictor.stream_query(q, top_k=top_k):
            # SSE format: "data: <content>\n\n"
            yield f"data: {token}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering on Render
        },
    )
