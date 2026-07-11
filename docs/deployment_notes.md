# Deployment Notes

This document captures decisions, gotchas, and operational details for hosting the assistant on Render (backend) and Vercel (frontend).

## Low-cost deployment (Groq plus OpenAI embeddings)

Goal: a public demo with free intent classification and generation. The deployed system uses **Groq's free tier** for both calls. Embeddings stay on OpenAI's `text-embedding-3-small`, so the current `chroma_db` and retrieval evaluation remain valid.

**Cost after this change:** intent classification and generation use Groq's free plan. The only paid runtime call is the OpenAI embedding for questions routed to synthesis. `text-embedding-3-small` is $0.02 per million input tokens. Fact lookups, clarifications, and health checks do not call OpenAI. To remove the OpenAI dependency, set `EMBEDDING_PROVIDER=local` and re-ingest. That changes the vector space and requires a corpus rebuild and new evaluation run.

Production call map:

| Query path | Groq | OpenAI |
|---|---|---|
| Verified fact lookup | Classifier | None |
| Clarification | Classifier | None |
| Synthesized answer | Classifier + generator | Query embedding |
| Health check | None | None |

GPT-4o-mini was benchmarked as an alternative intent classifier. It scored 31/33 versus Groq's 29/33, but the confidence intervals overlap. Groq was selected because it avoids classifier cost. The Groq model ID `openai/gpt-oss-20b` names the model served by Groq; it does not send the request to the OpenAI API.

**Accounts to create:**

1. **Groq:** [console.groq.com](https://console.groq.com) → sign in with Google/GitHub → *API Keys* → create key. Free, no credit card. This is the only new account required.
2. **OpenAI:** still needed for query embeddings. Set a hard usage cap (~$5/mo) in the dashboard as a safety net.
3. GitHub, Render, and Vercel: as before.

**Config:** `LLM_PROVIDER=groq`, `GROQ_API_KEY` (secret), `GROQ_MODEL=openai/gpt-oss-20b`, `EMBEDDING_PROVIDER=openai`, `OPENAI_API_KEY` (secret), `USE_RERANKER=false`, `ROUTER_ENABLED=true`, `INTENT_CLASSIFIER=zero_shot`, `INTENT_PROVIDER=groq`, and `VALIDATOR_MODE=flag`.

**Why `USE_RERANKER=false` in production:** the cross-encoder reranker pulls in torch (~700 MB) and won't fit Render's 512 MB free tier. The ablation already showed it didn't improve results, so dense + BM25 hybrid is the deployed config. `requirements.txt` omits `sentence-transformers` for the same reason.

**Model note:** Groq rotates model IDs (the older `llama-3.3-70b-versatile` is being retired). If a query returns "model not found," pick a current ID from [console.groq.com/docs/models](https://console.groq.com/docs/models) and set `GROQ_MODEL`. `openai/gpt-oss-120b` is the higher-quality option; `openai/gpt-oss-20b` is faster and lighter.

**Free-tier limits:** Groq free tier is roughly 30 requests/min and ~1,000/day. That covers portfolio demo traffic. Production traffic would need a different limit.

## Backend: Render

Blueprint file: `render.yaml`

Required environment variables:

| Variable | Required | Notes |
|----------|----------|-------|
| `LLM_PROVIDER` | No | `groq` (free, recommended) / `openai` (paid) / `ollama` (local dev). Defaults to `openai`. |
| `GROQ_API_KEY` | Yes if `LLM_PROVIDER=groq` | Secret. Free key from console.groq.com. |
| `GROQ_MODEL` | No | Defaults to `openai/gpt-oss-20b`. Any current ID from console.groq.com/docs/models. |
| `EMBEDDING_PROVIDER` | No | `openai` (default) or `local`. `local` requires re-ingesting the corpus. |
| `OPENAI_API_KEY` | Yes | Secret. Used for synthesized-path query embeddings; for generation only if `LLM_PROVIDER=openai`. |
| `USE_RERANKER` | No | Defaults `true`. Set `false` on Render free tier (avoids torch / OOM). |
| `USE_HYBRID` | No | Defaults `true`. Dense + BM25 fusion. |
| `TOP_K` | No | Defaults `4`. Chunks retrieved per query. |
| `CHROMA_PERSIST_PATH` | No | Defaults to `./chroma_db`. |
| `FRONTEND_ORIGIN` | Yes after frontend deploy | Public URL of the deployed Vercel frontend (used for CORS). |
| `ROUTER_ENABLED` | No | Set `true` for V2 routing. Defaults to `true`. |
| `INTENT_CLASSIFIER` | No | Set `zero_shot` for the selected production classifier. |
| `INTENT_PROVIDER` | No | Set `groq` for the cost-free production classifier. |
| `FACTS_DIR` | No | Defaults to `./data/facts`; the deployed build should load 88 facts. |
| `VALIDATOR_MODE` | No | `flag` is the production default; `strict` refuses hard citation or numeric failures. |

### Vector store on Render

`chroma_db/` is gitignored and is not present on the Render instance by default. Options for production:

1. Run ingestion at deploy time. Add a build-time step that downloads PDFs from a known bucket and runs `python -m src.ingestion --reset`. Simple, but inflates build time and re-embeds on every deploy.
2. Persist the vector store on a Render disk and ingest once. Cheapest at request time, requires a paid disk.
3. Move to a hosted vector store (Qdrant Cloud, Pinecone, etc.) for production. This is a future option; no hosted vector store is configured.

For a public launch, the practical path is option 1 with a small public corpus. Because the corpus here is small and static, the prebuilt `chroma_db/` is committed to the repo and ships with the deploy. No build-time embedding is needed. Interpretation queries call OpenAI with the same `text-embedding-3-small` model that built the store, so the query and stored vectors remain compatible.

### Health and readiness

`/health` returns degraded mode if `chroma_db/` is empty. The frontend shows a degraded-state badge instead of a hard error.

## Frontend: Vercel

Project root: `frontend/`

The frontend reads its API base URL from `frontend/config.js`, which exposes `window.RAG_CONFIG.API_BASE`. To point the deployed frontend at the deployed backend:

1. Update `frontend/config.js` with the Render API URL before deploying, or
2. Inject `window.RAG_CONFIG = { API_BASE: "..." }` via Vercel project settings using a custom HTML snippet.

A null/missing value falls back to `http://localhost:8000` so local development still works.

## CORS

`api/main.py` reads `FRONTEND_ORIGIN` from the environment and merges it with the localhost dev origins. After deploying the Vercel frontend, set this env var in Render to both origins (comma-separated): `FRONTEND_ORIGIN=https://rag.alvinalias.com,https://rag-engineering-assistant.vercel.app`, then redeploy. There is no hardcoded production URL in the source.

## Cost Ceiling

Render free tier and OpenAI usage together are bounded by:

- Render free instance: spins down on inactivity. First request after idle is slow.
- OpenAI: protect with a usage cap in the OpenAI dashboard. Recommended hard ceiling: $5/month for public demo traffic.

## Operational Checks

After every deploy, verify:

- `GET /health` returns `chroma_loaded: true`, `router_enabled: true`, and `facts_loaded: 88`.
- `POST /query` with a known in-corpus question returns a grounded answer with citations.
- Frontend network tab shows requests going to the Render URL, not localhost.
- Browser console is free of CORS errors.
