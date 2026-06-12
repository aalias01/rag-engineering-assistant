# Deployment Notes

This document captures decisions, gotchas, and operational details for hosting the assistant on Render (backend) and Vercel (frontend).

## Backend — Render

Blueprint file: `render.yaml`

Required environment variables:

| Variable | Required | Notes |
|----------|----------|-------|
| `OPENAI_API_KEY` | Yes | Secret. Set in the Render dashboard, never commit. |
| `LLM_PROVIDER` | No | Defaults to `openai`. Set to `ollama` only for local runs. |
| `CHROMA_PERSIST_PATH` | No | Defaults to `./chroma_db`. |
| `FRONTEND_ORIGIN` | Yes after frontend deploy | Public URL of the deployed Vercel frontend (used for CORS). |

### Vector store on Render

`chroma_db/` is gitignored and is not present on the Render instance by default. Options for production:

1. Run ingestion at deploy time. Add a build-time step that downloads PDFs from a known bucket and runs `python -m src.ingestion --reset`. Simple, but inflates build time and re-embeds on every deploy.
2. Persist the vector store on a Render disk and ingest once. Cheapest at request time, requires a paid disk.
3. Move to a hosted vector store (Qdrant Cloud, Pinecone, etc.) for production. Documented as the future path, not implemented.

For the portfolio launch, the practical path is option 1 with a small public corpus.

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
- OpenAI — protect with a usage cap in the OpenAI dashboard. Recommended hard ceiling: $5/month for portfolio traffic.

## Operational Checks

After every deploy, verify:

- `GET /health` returns `chroma_loaded: true` and the expected `collection_size`.
- `POST /query` with a known in-corpus question returns a grounded answer with citations.
- Frontend network tab shows requests going to the Render URL, not localhost.
- Browser console is free of CORS errors.
