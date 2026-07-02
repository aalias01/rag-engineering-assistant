# Deployment Notes

This document captures decisions, gotchas, and operational details for hosting the assistant on Render (backend) and Vercel (frontend).

## Zero-cost deployment (Groq for generation)

Goal: a public demo that costs ~$0 to run. The expensive part of a RAG query is the LLM that writes the answer, not the vector search — so we keep everything as built and swap only the generator to **Groq's free tier** (OpenAI-compatible, no credit card). Embeddings stay on OpenAI's `text-embedding-3-small`, which means the current `chroma_db` is unchanged and the retrieval eval numbers stay valid.

**Cost after this change:** generation is free (Groq). The only paid call left is embedding each incoming query with OpenAI, at ~$0.02 per *million* tokens — a question is ~20 tokens, so realistically a few cents even under heavy demo traffic. Effectively free, but not literally $0. To reach a true $0 with no OpenAI account at all, set `EMBEDDING_PROVIDER=local` and re-ingest: that swaps in `all-MiniLM-L6-v2`, which changes the vector space, so the corpus must be rebuilt and the eval re-run.

**Accounts to create:**

1. **Groq** — [console.groq.com](https://console.groq.com) → sign in with Google/GitHub → *API Keys* → create key. Free, no credit card. This is the only new account required.
2. **OpenAI** — still needed for query embeddings. Set a hard usage cap (~$5/mo) in the dashboard as a safety net.
3. GitHub, Render, Vercel — as before.

**Config (already wired into `render.yaml`):** `LLM_PROVIDER=groq`, `GROQ_API_KEY` (secret), `GROQ_MODEL=openai/gpt-oss-20b`, `EMBEDDING_PROVIDER=openai`, `OPENAI_API_KEY` (secret), `USE_RERANKER=false`.

**Why `USE_RERANKER=false` in production:** the cross-encoder reranker pulls in torch (~700 MB) and won't fit Render's 512 MB free tier. The ablation already showed it didn't improve results, so dense + BM25 hybrid is the deployed config. `requirements.txt` omits `sentence-transformers` for the same reason.

**Model note:** Groq rotates model IDs (the older `llama-3.3-70b-versatile` is being retired). If a query returns "model not found," pick a current ID from [console.groq.com/docs/models](https://console.groq.com/docs/models) and set `GROQ_MODEL`. `openai/gpt-oss-120b` is the higher-quality option; `openai/gpt-oss-20b` is faster and lighter.

**Free-tier limits:** Groq free tier is roughly 30 requests/min and ~1,000/day — plenty for a portfolio demo, not a production load.

## Backend — Render

Blueprint file: `render.yaml`

Required environment variables:

| Variable | Required | Notes |
|----------|----------|-------|
| `LLM_PROVIDER` | No | `groq` (free, recommended) / `openai` (paid) / `ollama` (local dev). Defaults to `openai`. |
| `GROQ_API_KEY` | Yes if `LLM_PROVIDER=groq` | Secret. Free key from console.groq.com. |
| `GROQ_MODEL` | No | Defaults to `openai/gpt-oss-20b`. Any current ID from console.groq.com/docs/models. |
| `EMBEDDING_PROVIDER` | No | `openai` (default) or `local`. `local` requires re-ingesting the corpus. |
| `OPENAI_API_KEY` | Yes | Secret. Used for query embeddings always; for generation only if `LLM_PROVIDER=openai`. |
| `USE_RERANKER` | No | Defaults `true`. Set `false` on Render free tier (avoids torch / OOM). |
| `USE_HYBRID` | No | Defaults `true`. Dense + BM25 fusion. |
| `TOP_K` | No | Defaults `4`. Chunks retrieved per query. |
| `CHROMA_PERSIST_PATH` | No | Defaults to `./chroma_db`. |
| `FRONTEND_ORIGIN` | Yes after frontend deploy | Public URL of the deployed Vercel frontend (used for CORS). |

### Vector store on Render

`chroma_db/` is gitignored and is not present on the Render instance by default. Options for production:

1. Run ingestion at deploy time. Add a build-time step that downloads PDFs from a known bucket and runs `python -m src.ingestion --reset`. Simple, but inflates build time and re-embeds on every deploy.
2. Persist the vector store on a Render disk and ingest once. Cheapest at request time, requires a paid disk.
3. Move to a hosted vector store (Qdrant Cloud, Pinecone, etc.) for production. Documented as the future path, not implemented.

For a public launch, the practical path is option 1 with a small public corpus. Because the corpus here is small and static, the simplest variant is to commit the prebuilt `chroma_db/` to the repo (remove it from `.gitignore`) so it ships with the deploy — no re-embedding at build time. Query-time embedding still calls OpenAI with the same `text-embedding-3-small` model that built the store, so the store stays compatible.

### Health and readiness

`/health` returns degraded mode if `chroma_db/` is empty. The frontend's badge surfaces this as "Degraded — no documents ingested" rather than a hard error.

## Frontend — Vercel

Project root: `frontend/`

The frontend reads its API base URL from `frontend/config.js`, which exposes `window.RAG_CONFIG.API_BASE`. To point the deployed frontend at the deployed backend:

1. Update `frontend/config.js` with the Render API URL before deploying, or
2. Inject `window.RAG_CONFIG = { API_BASE: "..." }` via Vercel project settings using a custom HTML snippet.

A null/missing value falls back to `http://localhost:8000` so local development still works.

## CORS

`api/main.py` reads `FRONTEND_ORIGIN` from the environment and merges it with the localhost dev origins. After deploying the Vercel frontend, set this env var in Render to the production URL (e.g. `https://rag-engineering-assistant.vercel.app`) and redeploy. There is no hardcoded production URL in the source.

## Cost Ceiling

Render free tier and OpenAI usage together are bounded by:

- Render free instance — spins down on inactivity. First request after idle is slow.
- OpenAI — protect with a usage cap in the OpenAI dashboard. Recommended hard ceiling: $5/month for public demo traffic.

## Operational Checks

After every deploy, verify:

- `GET /health` returns `chroma_loaded: true` and the expected `collection_size`.
- `POST /query` with a known in-corpus question returns a grounded answer with citations.
- Frontend network tab shows requests going to the Render URL, not localhost.
- Browser console is free of CORS errors.
