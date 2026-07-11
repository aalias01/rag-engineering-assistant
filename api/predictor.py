"""
api/predictor.py — Retriever + Generator wiring for the FastAPI layer.

Loads the ChromaDB collection and initialises the Retriever/Generator once
at API startup (via FastAPI lifespan), then serves requests from cached state.

v2 routing (ROUTER_ENABLED, default true): every query is intent-classified
first (src/router.py). "lookup" intents try the typed facts DB
(src/facts.py) — a deterministic, LLM-free path; on a miss they fall through
to the v1 retrieve+generate path, which now runs the post-generation citation
validator (src/validator.py). "clarify" intents return a clarification
question instead of a guess. Any router/facts failure falls open to the v1
path — v2 can only add capability, never subtract.

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
_router = None
_facts_db = None


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable (returns default when unset)."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _router_enabled() -> bool:
    return _env_bool("ROUTER_ENABLED", True)


def _load_routing() -> None:
    """Initialise the intent router + facts DB (v2). Failure is non-fatal:
    the API simply behaves as v1."""
    global _router, _facts_db
    try:
        from src.facts import FactsDB
        from src.router import IntentRouter

        _router = IntentRouter()
        _facts_db = FactsDB(os.getenv("FACTS_DIR", "./data/facts"))
        print(
            f"Router loaded: backend={_router.backend}, "
            f"facts={len(_facts_db)} across "
            f"{len({f.source_doc for f in _facts_db.facts})} source docs."
        )
    except Exception as e:
        _router = None
        _facts_db = None
        print(f"WARNING: v2 router/facts unavailable — running v1 behavior.\n{e}")


def load_all() -> None:
    """
    Initialise the retriever and generator. Called once at API startup
    via FastAPI lifespan. Sets _chroma_loaded = True on success.
    """
    global _retriever, _generator, _collection_size, _chroma_loaded

    if _router_enabled():
        _load_routing()

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

        use_hybrid = _env_bool("USE_HYBRID", True)
        use_reranker = _env_bool("USE_RERANKER", True)
        top_k = int(os.getenv("TOP_K", "4"))
        _retriever = Retriever(use_hybrid=use_hybrid, use_reranker=use_reranker, top_k=top_k)
        _generator = Generator()
        _chroma_loaded = True
        print(
            f"RAG system loaded. Collection: {_collection_size} chunks. "
            f"Retriever: hybrid={use_hybrid}, reranker={use_reranker}, top_k={top_k}. "
            f"LLM provider: {os.getenv('LLM_PROVIDER', 'openai')}."
        )

    except Exception as e:
        print(f"WARNING: Could not load RAG system — API running in degraded mode.\n{e}")


def is_ready() -> bool:
    return _chroma_loaded and _retriever is not None and _generator is not None


def _request_retriever(
    top_k: int,
    use_hybrid: bool,
    use_reranker: bool,
):
    from src.retriever import Retriever

    # Memory ceiling: on constrained hosts (e.g. Render free tier, 512 MB) the
    # cross-encoder reranker is disabled via USE_RERANKER=false because it pulls
    # in torch (~700 MB). Honor that here so a per-request use_reranker=True
    # can't load torch and OOM the box. (Ablation showed reranking didn't help.)
    if not _env_bool("USE_RERANKER", True):
        use_reranker = False
    if not _env_bool("USE_HYBRID", True):
        use_hybrid = False

    # Reuse the cached retriever unless the request overrides its config.
    if (use_hybrid, use_reranker, top_k) == (
        _retriever.use_hybrid,
        _retriever.use_reranker,
        _retriever.top_k,
    ):
        return _retriever

    return Retriever(use_hybrid=use_hybrid, use_reranker=use_reranker, top_k=top_k)


_GENERIC_CLARIFICATION = (
    "Could you make the question more specific? For value lookups, name the "
    "item and its context — for example the product class and region for an "
    "efficiency standard, or the chemical name for a threshold quantity. For "
    "conceptual questions, one full sentence about what you want explained "
    "works best."
)


def _route(q: str) -> tuple[str, dict, object | None]:
    """
    Classify the query and attempt the deterministic paths.

    Returns (path, meta, payload):
        path "lookup"    — payload is the matched Fact
        path "clarify"   — payload is the clarification string
        path "interpret" — payload is None (caller runs retrieve+generate)
    """
    meta: dict = {"intent": None, "intent_method": None}
    if not (_router_enabled() and _router and _facts_db):
        return "interpret", meta, None
    try:
        decision = _router.classify(q)
        meta["intent"] = decision.intent
        meta["intent_method"] = decision.method

        if decision.intent == "lookup":
            result = _facts_db.lookup(q)
            if result.status == "hit":
                return "lookup", meta, result.fact
            if result.status == "ambiguous":
                return "clarify", meta, result.clarification
            return "interpret", meta, None  # miss → fall through (fail open)

        if decision.intent == "clarify":
            # A lookup-shaped but underspecified query may still match several
            # facts — use their qualifiers to ask a *specific* question.
            result = _facts_db.lookup(q)
            if result.status == "ambiguous":
                return "clarify", meta, result.clarification
            return "clarify", meta, _GENERIC_CLARIFICATION

        return "interpret", meta, None
    except Exception as e:
        print(f"WARNING: routing failed, using interpret path. {e}")
        return "interpret", meta, None


def _lookup_response(q: str, fact) -> dict:
    """Deterministic response envelope for a facts-DB hit. No LLM involved."""
    from src.facts import render_fact_answer

    return {
        "answer": render_fact_answer(fact),
        "sources": [{"source": fact.source_doc, "page": fact.source_page}],
        "chunks": [
            {
                "text": fact.quote,
                "source": fact.source_doc,
                "page": fact.source_page,
                "retrieval_method": "facts_db",
                "rrf_score": None,
                "rerank_score": None,
            }
        ],
        "chunks_used": 1,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "model": "facts_db",
        "provider": "deterministic",
        "refused": False,
        "route": "factual_lookup",
        "fact_id": fact.fact_id,
        "fact_status": fact.curation_status,
        "validation": None,
        "clarification": None,
    }


def _clarify_response(q: str, clarification: str) -> dict:
    """Deterministic response envelope for the clarify path."""
    return {
        "answer": clarification,
        "sources": [],
        "chunks": [],
        "chunks_used": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "model": "router",
        "provider": "deterministic",
        "refused": False,
        "route": "clarification",
        "validation": None,
        "clarification": clarification,
    }


def query(
    q: str,
    top_k: int = 4,
    use_hybrid: bool = True,
    use_reranker: bool = True,
) -> dict:
    """
    Route the query, then run the matching path.

    lookup    → templated answer from the typed facts DB (no LLM)
    clarify   → clarification question (no retrieval, no LLM)
    interpret → v1 retrieval + generation + post-generation citation validation

    Returns the response dict (see _lookup_response for the added v2 keys).
    Raises RuntimeError if the system is not loaded.
    """
    if not is_ready():
        raise RuntimeError(
            "RAG system not ready. Documents may not be ingested yet. "
            "Run `python -m src.ingestion` to ingest PDFs into ChromaDB."
        )

    path, meta, payload = _route(q)
    if path == "lookup":
        result = _lookup_response(q, payload)
        result.update(meta)
        return result
    if path == "clarify":
        result = _clarify_response(q, payload)
        result.update(meta)
        return result

    retriever = _request_retriever(top_k, use_hybrid, use_reranker)
    chunks = retriever.retrieve(q, top_k=top_k)
    result = _generator.generate(q, chunks)
    result["chunks"] = chunks
    result["route"] = "synthesized"
    result["clarification"] = None
    result.update(meta)

    # Post-generation guardrail (v2): verify citations point at retrieved
    # chunks and numbers appear in them. VALIDATOR_MODE=strict converts hard
    # failures into refusals; default "flag" annotates for the frontend badge.
    try:
        from src.generator import REFUSAL_PHRASE
        from src.validator import apply_policy, validate_answer

        report = validate_answer(result["answer"], chunks)
        answer, report = apply_policy(result["answer"], report, REFUSAL_PHRASE)
        result["answer"] = answer
        if report.status == "refused":
            result["refused"] = True
        result["validation"] = report.to_dict()
    except Exception as e:
        print(f"WARNING: citation validation failed open. {e}")
        result["validation"] = None
    return result


async def stream_query(
    q: str,
    top_k: int = 4,
    use_hybrid: bool = True,
    use_reranker: bool = True,
):
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

    import json as _json

    path, meta, payload = _route(q)

    if path in ("lookup", "clarify"):
        # Deterministic paths: emit the templated answer word-by-word (keeps
        # the frontend's streaming UX), then the metadata frame.
        result = _lookup_response(q, payload) if path == "lookup" else _clarify_response(q, payload)
        result.update(meta)
        for word in result["answer"].split(" "):
            yield word + " "
        metadata = {k: v for k, v in result.items() if k != "answer"}
        yield f"__METADATA__:{_json.dumps(metadata)}"
        return

    retriever = _request_retriever(top_k, use_hybrid, use_reranker)
    chunks = retriever.retrieve(q, top_k=top_k)

    # Interpret path: pass tokens through, but intercept the generator's
    # metadata frame to append route info + the validation report. Note the
    # streaming limitation: tokens are already on the wire, so strict mode
    # cannot withhold a streamed answer — validation is advisory here (the
    # frontend shows a warning badge). The blocking POST /query endpoint is
    # the enforcing path.
    answer_parts: list[str] = []
    async for token in _generator.stream(q, chunks):
        if token.startswith("__METADATA__:"):
            try:
                metadata = _json.loads(token.replace("__METADATA__:", "", 1))
            except _json.JSONDecodeError:
                yield token
                continue
            metadata["route"] = "synthesized"
            metadata["clarification"] = None
            metadata.update(meta)
            try:
                from src.validator import validate_answer

                report = validate_answer("".join(answer_parts), chunks)
                metadata["validation"] = report.to_dict()
            except Exception as e:
                print(f"WARNING: citation validation failed open. {e}")
                metadata["validation"] = None
            yield f"__METADATA__:{_json.dumps(metadata)}"
        else:
            answer_parts.append(token)
            yield token


def get_status() -> dict:
    return {
        "chroma_loaded": _chroma_loaded,
        "collection_size": _collection_size,
        "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
        "router_enabled": bool(_router_enabled() and _router and _facts_db),
        "intent_classifier": _router.backend if _router else None,
        "facts_loaded": len(_facts_db) if _facts_db else 0,
    }
