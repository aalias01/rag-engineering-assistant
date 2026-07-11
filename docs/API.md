# API reference

The FastAPI service exposes readiness, blocking query, and Server-Sent Events (SSE) endpoints. Interactive OpenAPI documentation is available at [`/docs`](https://rag-engineering-assistant-api.onrender.com/docs).

## `GET /health`

Returns API and retrieval-store readiness.

```json
{
  "status": "healthy",
  "chroma_loaded": true,
  "collection_size": 2091,
  "version": "2.0.0",
  "llm_provider": "groq",
  "router_enabled": true,
  "intent_classifier": "zero_shot",
  "facts_loaded": 88
}
```

`status` is `degraded` when the vector store is unavailable. The frontend treats transport failures as a sleeping free-tier service, shows wake-up copy, and retries this endpoint every 10 seconds.

## `POST /query`

Runs the router and returns a complete JSON response.

### Request

```json
{
  "query": "How often must a process hazard analysis be updated and revalidated under the PSM standard?",
  "top_k": 4,
  "use_hybrid": true,
  "use_reranker": false
}
```

| Field | Type | Default | Notes |
|---|---|---:|---|
| `query` | string | required | 5–1,000 characters |
| `top_k` | integer | `4` | 1–10 |
| `use_hybrid` | boolean | `true` | Adds BM25 + Reciprocal Rank Fusion |
| `use_reranker` | boolean | `true` | Adds the cross-encoder reranker; the deployed frontend sends `false` |

### Synthesized response

```json
{
  "query": "...",
  "answer": "...",
  "sources": [{"source": "document.pdf", "page": 12}],
  "chunks_used": 4,
  "chunks": [],
  "prompt_tokens": 1200,
  "completion_tokens": 180,
  "cost_usd": 0.0,
  "latency_ms": 2400,
  "model": "openai/gpt-oss-20b",
  "provider": "groq",
  "refused": false,
  "route": "synthesized",
  "intent": "interpret",
  "intent_method": "zero_shot",
  "validation": {
    "status": "passed",
    "citations_valid": 2,
    "citations_found": 2,
    "invalid_citations": [],
    "ungrounded_numbers": []
  }
}
```

`route` is one of:

- `factual_lookup`: a deterministic answer assembled from a verified facts file; also returns `fact_id` and `fact_status`.
- `synthesized`: retrieval plus LLM generation and post-generation validation.
- `clarification`: an underspecified lookup; returns the follow-up in `clarification` without retrieval or generation.

Pre-v2 clients can ignore the routing and validation fields.

## `GET /query/stream`

Streams answer tokens over SSE.

```text
/query/stream?q=your%20question&top_k=4&use_hybrid=true&use_reranker=false
```

Ordinary `data:` frames contain answer tokens. The final frame starts with `__METADATA__:` and contains the same sources, chunks, route, validation, timing, and usage metadata returned by the blocking endpoint.

Because already-sent SSE tokens cannot be withdrawn, `VALIDATOR_MODE=strict` can replace an invalid answer on `POST /query`, while streaming reports the validation result in the final metadata frame for the frontend badge.

## Errors and refusal

- Validation errors use FastAPI's normal `422` response.
- Runtime failures return a non-2xx response with a `detail` field.
- Out-of-corpus questions return a normal response with `refused: true` and the configured refusal phrase; refusal is a model behavior, not an HTTP error.

## Local URLs

- Docs: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>
- OpenAPI JSON: <http://localhost:8000/openapi.json>
