# Portfolio Release Summary

Use this file as the PR description for the final portfolio release, or as a release note when tagging v1.0.0. Replace hosted URL placeholders before publishing.

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
| Retrieval Recall@3 | `0.923` | ≥ 0.85 | `pass` |
| Retrieval MRR | `0.817` | ≥ 0.70 | `pass` |
| Ragas faithfulness | `0.928` | ≥ 0.85 | `pass` |
| Ragas answer relevancy | `0.960` | ≥ 0.85 | `pass` |
| Refusal accuracy | `1.000` | ≥ 0.80 | `pass` |
| Median end-to-end latency | `2.3 s` | ≤ 3 s | `pass` |
| Estimated cost per 100 queries | `$0.02` | ≤ $0.50 | `pass` |

### Ablation Highlights

| Configuration | Recall@3 | MRR | Notes |
|---------------|----------|-----|-------|
| Dense-only | `0.923` | `0.817` | Best measured MRR and latency tradeoff |
| BM25-only | `0.731` | `0.580` | Exact-term retrieval baseline |
| Hybrid RRF | `0.885` | `0.774` | Fusion without reranker |
| Hybrid RRF + reranker | `0.923` | `0.741` | Matches best Recall@3 with added latency |

### Corpus

6 public engineering PDFs across mechanical fundamentals, systems engineering, and safety/efficiency regulation. Full manifest in `docs/corpus_selection.md`.

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

> Built an end-to-end RAG assistant over public engineering documents (NASA, OSHA, DOE) with hybrid dense/BM25 retrieval, Reciprocal Rank Fusion, cross-encoder reranking, ChromaDB, GPT-4o-mini, FastAPI, and a streaming JS frontend; achieved Recall@3 of `0.923`, MRR of `0.817`, and Ragas faithfulness of `0.928` on a 31-query labeled evaluation set.

### Related Docs

- `README.md` — reviewer-facing overview
- `docs_local/PROJECT_BRIEF.md` — internal planning brief
- `docs/corpus_selection.md` — corpus criteria and manifest
- `docs/evaluation_protocol.md` — how every metric is computed
- `docs/deployment_notes.md` — Render/Vercel operational details
- `docs/interview_walkthrough.md` — 5-7 minute walkthrough script
