# Portfolio Release Summary

Use this file as the PR description for the final portfolio release, or as a release note when tagging v1.0.0. Replace bracketed placeholders before publishing.

---

## RAG Engineering Assistant — v1.0.0

End-to-end retrieval-augmented generation system over a curated corpus of public engineering documents. Built as a portfolio-quality demonstration of practical GenAI engineering: ingestion, hybrid retrieval, reranking, citation-grounded prompting, evaluation, API, and a usable chat UI.

### Live Demo

- Frontend: `[https://<your-app>.vercel.app]`
- API: `[https://<your-render-service>.onrender.com]`
- API docs: `[https://<your-render-service>.onrender.com/docs]`

### What's In This Release

- PDF ingestion with page-level metadata (PyMuPDF + pdfplumber fallback) and hash-based dedupe.
- ChromaDB persistent vector store with OpenAI `text-embedding-3-small` (local MiniLM optional).
- Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion, with optional cross-encoder reranker.
- Grounded generation with GPT-4o-mini and a strict citation-required prompt (Ollama path supported).
- FastAPI backend with health, blocking query, and SSE streaming endpoints.
- Vanilla-JS frontend with streaming, source pills, retrieved-chunk inspector, and latency/cost stats.
- Hand-labeled evaluation set (~30 queries) with retrieval (Recall@3, MRR) and generation (Ragas faithfulness, answer relevancy, refusal accuracy) metrics.
- Ablations across chunk size, retrieval mode, and reranker on/off.

### Measured Results

| Metric | Value | Target | Pass |
|--------|-------|--------|------|
| Retrieval Recall@3 | `[X.XX]` | ≥ 0.85 | `[ ]` |
| Retrieval MRR | `[X.XX]` | ≥ 0.70 | `[ ]` |
| Ragas faithfulness | `[X.XX]` | ≥ 0.85 | `[ ]` |
| Ragas answer relevancy | `[X.XX]` | ≥ 0.85 | `[ ]` |
| Refusal accuracy | `[X.XX]` | ≥ 0.80 | `[ ]` |
| Median end-to-end latency | `[X.X s]` | ≤ 3 s | `[ ]` |
| Estimated cost per 100 queries | `[$X.XX]` | ≤ $0.50 | `[ ]` |

### Ablation Highlights

| Configuration | Recall@3 | MRR | Notes |
|---------------|----------|-----|-------|
| Dense-only | `[X.XX]` | `[X.XX]` | Baseline |
| BM25-only | `[X.XX]` | `[X.XX]` | Exact-term retrieval |
| Hybrid RRF | `[X.XX]` | `[X.XX]` | Final candidate |
| Hybrid RRF + reranker | `[X.XX]` | `[X.XX]` | Winning config |

### Corpus

`[N]` public engineering PDFs across `[domains]`. Full manifest in `docs/corpus_selection.md`.

### Known Limitations

- Scanned PDFs require OCR; out of scope for this release.
- BM25 tokenizer is whitespace-only — punctuated identifiers may still need normalization.
- Render free tier may evict the cross-encoder reranker under memory pressure.
- Streaming endpoint estimates completion tokens unless provider usage metadata is available.

### Out Of Scope

- Multi-tenant access control
- Document re-ingestion at scale (one-shot ingest only)
- Hosted vector store (ChromaDB local persisted only)

### How To Reproduce

```bash
git clone https://github.com/aalias01/rag-engineering-assistant
cd rag-engineering-assistant
conda env create -f environment.yml && conda activate rag-assistant
cp .env.example .env  # add OPENAI_API_KEY
# Place the corpus PDFs in data/documents/
python -m src.ingestion --reset
python scripts/validate_eval_set.py
python -m src.eval --retrieval --k 3
python -m src.eval --ragas
uvicorn api.main:app --reload
```

### Resume Bullet

> Built an end-to-end RAG assistant over public engineering documents (NASA, OSHA, energy guides) with hybrid dense/BM25 retrieval, Reciprocal Rank Fusion, cross-encoder reranking, ChromaDB, GPT-4o-mini, FastAPI, and a streaming JS frontend; achieved Recall@3 of `[X.XX]`, MRR of `[X.XX]`, and Ragas faithfulness of `[X.XX]` on a 30-query labeled evaluation set.

### Related Docs

- `README.md` — reviewer-facing overview
- `docs_local/PROJECT_BRIEF.md` — internal planning brief
- `docs/corpus_selection.md` — corpus criteria and manifest
- `docs/evaluation_protocol.md` — how every metric is computed
- `docs/deployment_notes.md` — Render/Vercel operational details
- `docs/interview_walkthrough.md` — 5-7 minute walkthrough script
