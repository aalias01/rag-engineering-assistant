# RAG Engineering Assistant

> Query engineering standards in natural language and receive grounded answers with page-level source citations.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green)](https://www.langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal)](https://fastapi.tiangolo.com/)

RAG Engineering Assistant is an end-to-end retrieval-augmented generation system for technical engineering documents. It ingests PDF standards, manuals, and reports; retrieves relevant excerpts with hybrid search; reranks the best candidates; and generates concise answers grounded only in the retrieved source text.

The project is designed as a portfolio-quality demonstration of practical GenAI engineering: document ingestion, vector search, hybrid retrieval, reranking, citation-grounded prompting, evaluation, API deployment, and a usable chat interface.

## Current Status

The system is fully implemented and evaluated on a six-document public-domain engineering corpus (778 pages, 2,091 chunks). All evaluation targets are met — see [Results](#results).

| Area | Status | Notes |
|------|--------|-------|
| PDF ingestion | Complete | PyMuPDF extraction with pdfplumber fallback, hash-based reingestion skip |
| Corpus | Complete | 6 public-domain documents, 778 pages — see `docs/corpus_selection.md` |
| Vector store | Complete | ChromaDB persistent collection, 2,091 chunks at chunk size 300 (ablation-selected) |
| Retrieval | Complete | Dense retrieval, BM25 retrieval, Reciprocal Rank Fusion, optional cross-encoder reranking |
| Generation | Complete | GPT-4o-mini by default; Ollama local model path supported |
| API | Complete | FastAPI `GET /health`, `POST /query`, `GET /query/stream` |
| Frontend | Complete | Vanilla JS chat UI with streaming, citations, retrieved chunk panel, latency/cost stats |
| Evaluation set | Complete | 31 hand-labeled queries with verified page-level ground truth |
| Metrics | Complete | Recall@3 0.923 · MRR 0.817 · faithfulness 0.928 · relevancy 0.960 · refusal 1.000 |
| Deployment | Configured, not deployed | Render blueprint included; Vercel frontend needs deployed API URL |

Live demo and hosted API links should be added here after deployment:

- Live demo: pending
- API docs: pending

## Why This Project Matters

Engineers often spend a large share of their time searching standards, procedures, design manuals, and technical reports. Generic search can find documents, but it does not reliably explain which section answers the question or whether the answer is grounded in the source text.

This project focuses on the parts of RAG that matter in production:

- Retrieval quality is measured separately from answer quality.
- Hybrid retrieval handles engineering acronyms and exact specification names better than dense search alone.
- The generator is constrained to answer only from retrieved excerpts.
- Every response includes source document and page citations.
- Out-of-corpus questions are part of the evaluation plan, not an afterthought.

## The Corpus

Six public-domain engineering documents (778 pages), spanning mechanical fundamentals, systems engineering, and safety/efficiency regulation — full provenance in `docs/corpus_selection.md`:

| Document | Pages | Domain |
|----------|-------|--------|
| DOE-HDBK-1012 Vol 1 — Thermodynamics | 139 | Mechanical / HVAC fundamentals |
| DOE-HDBK-1012 Vol 2 — Heat Transfer | 80 | Mechanical / HVAC fundamentals |
| DOE-HDBK-1018 Vol 2 — Valves & Mechanical Components | 130 | Plant equipment |
| NASA Systems Engineering Handbook (SP-2016-6105 Rev 2) | 297 | Systems engineering |
| OSHA 3132 — Process Safety Management | 59 | Industrial safety regulation |
| DOE Final Rule 82 FR 1786 — Residential CAC/HP Efficiency Standards | 73 | Federal energy regulation |

Every document type here is one I worked with directly across 12 years in HVAC, subsea, and manufacturing engineering. The 2017 DOE final rule is the regulation behind the January 2023 product-line redesign I led at Rheem Manufacturing — this corpus is the problem I lived, not a demo dataset.

## Example Questions

```text
What is the difference between a relief valve and a safety valve?

How often must a process hazard analysis be updated and revalidated under the PSM standard?

What SEER level did the 2017 DOE final rule set for split-system air conditioners in the Southeast?

How does NASA define systems engineering?
```

## Architecture

```text
Engineering PDFs
    |
    v
PyMuPDF / pdfplumber extraction
    |
    v
Page-level text with source metadata
    |
    v
Recursive chunking, default about 300 tokens with 30-token overlap (ablation-selected)
    |
    v
Embeddings: OpenAI text-embedding-3-small or local MiniLM
    |
    v
ChromaDB persistent vector store
    |
    +---------------------------+
                                |
User query                      |
    |                           |
    +--> dense vector search ---+
    |
    +--> BM25 keyword search
              |
              v
Reciprocal Rank Fusion
              |
              v
Cross-encoder reranker
              |
              v
Top retrieved excerpts + query
              |
              v
GPT-4o-mini citation-grounded prompt
              |
              v
Answer, sources, retrieved chunks, latency, cost estimate
```

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| PDF processing | PyMuPDF, pdfplumber | Extract page text while preserving source/page metadata |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Produce overlapping retrieval units |
| Embeddings | OpenAI `text-embedding-3-small` | Default semantic retrieval model |
| Local embeddings | `all-MiniLM-L6-v2` | Offline option via sentence-transformers |
| Vector database | ChromaDB | Local persistent vector store |
| Keyword retrieval | rank-bm25 | Exact-term matching for standards, acronyms, and spec identifiers |
| Fusion | Reciprocal Rank Fusion | Combine dense and keyword rankings |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Improve top-result ordering |
| LLM | GPT-4o-mini | Grounded answer generation |
| Local LLM option | Ollama + Llama 3.1 8B | Zero-cost local generation path |
| API | FastAPI | Health checks, JSON queries, SSE streaming |
| Frontend | HTML, CSS, vanilla JS | Chat UI with sources and retrieved chunks |
| Evaluation | Custom metrics + Ragas | Recall@k, MRR, faithfulness, answer relevancy |

## Setup

```bash
git clone https://github.com/aalias01/rag-engineering-assistant
cd rag-engineering-assistant

conda env create -f environment.yml
conda activate rag-assistant

cp .env.example .env
```

Edit `.env` and set:

```text
OPENAI_API_KEY=your-openai-api-key
```

Then add PDF documents:

```bash
mkdir -p data/documents
# Place selected public or license-compatible engineering PDFs in data/documents/
```

Ingest documents:

```bash
python -m src.ingestion --reset
```

Run a smoke test:

```bash
python scripts/smoke_test.py
```

Start the API:

```bash
uvicorn api.main:app --reload
```

Open:

- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Frontend: open `frontend/index.html` in a browser

The frontend reads its API base URL from `frontend/config.js` (`window.RAG_CONFIG.API_BASE`). It defaults to `http://localhost:8000`. Change `config.js` (or inject `window.RAG_CONFIG` via Vercel settings) when pointing at a deployed backend — no edit to `frontend/app.js` is needed.

## API

### `GET /health`

Returns system readiness, ChromaDB load status, collection size, and active LLM provider.

### `POST /query`

Example request:

```json
{
  "query": "What does OSHA 1910.217 require for mechanical power press guarding?",
  "top_k": 4,
  "use_hybrid": true,
  "use_reranker": true
}
```

Example response shape:

```json
{
  "query": "...",
  "answer": "...",
  "sources": [{"source": "document.pdf", "page": 12}],
  "chunks_used": 4,
  "chunks": [],
  "prompt_tokens": 1200,
  "completion_tokens": 180,
  "cost_usd": 0.00029,
  "latency_ms": 2400,
  "model": "gpt-4o-mini",
  "provider": "openai"
}
```

### `GET /query/stream`

Streams answer tokens with Server-Sent Events:

```text
/query/stream?q=your%20question&top_k=4
```

## Evaluation

The evaluation set (`data/eval/test_queries.jsonl`) contains 31 hand-labeled queries with verified page-level ground truth: 21 in-corpus questions, 5 borderline questions requiring synthesis across chunks, and 5 out-of-corpus questions that must be refused — including deliberately hard refusal traps (e.g. a SEER2/2029 question that shares nearly all its vocabulary with an in-corpus document).

Retrieval metrics (Recall@3, MRR) are computed separately from generation metrics (Ragas faithfulness and answer relevancy via gpt-4o-mini judging, plus refusal accuracy). The full suite runs via `python scripts/run_full_eval.py` and writes `eval_results/full_results.json`.

## Repository Structure

```text
rag-engineering-assistant/
├── README.md
├── .env.example
├── environment.yml
├── requirements.txt
├── requirements-dev.txt        # adds ragas/datasets for evaluation
├── render.yaml
├── runtime.txt
├── api/
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
├── data/
│   ├── documents/              # gitignored PDF corpus
│   └── eval/
│       └── test_queries.jsonl
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── notebooks/
│   ├── 01_document_processing.ipynb
│   ├── 02_retrieval_evaluation.ipynb
│   ├── 03_rag_pipeline.ipynb
│   └── 04_reranker_ablation.ipynb
├── eval_results/
│   └── full_results.json       # measured metrics (committed for reproducibility)
├── scripts/
│   ├── smoke_test.py
│   ├── validate_eval_set.py
│   ├── run_full_eval.py        # one-shot evaluation suite
│   ├── finalize_chunk300.py
│   └── capture_demo.md
├── docs/
│   ├── corpus_selection.md
│   ├── evaluation_protocol.md
│   ├── deployment_notes.md
│   ├── interview_walkthrough.md
│   └── portfolio_pr_summary.md
├── LICENSE
└── src/
    ├── ingestion.py
    ├── retriever.py
    ├── generator.py
    └── eval.py
```

## Deployment

### Backend on Render

1. Push the repository to GitHub.
2. Create a Render Blueprint from this repo.
3. Render reads `render.yaml`.
4. Add `OPENAI_API_KEY` as a secret environment variable.
5. After deploy, open `/health` and `/docs`.

### Frontend on Vercel

1. Create a Vercel project from the same repo.
2. Set the project root to `frontend/`.
3. Update `API_BASE` in `frontend/config.js` to the Render API URL.
4. In Render, set `FRONTEND_ORIGIN` to the Vercel URL (CORS is read from env, not hardcoded).
5. Redeploy the backend.

## Results

All metrics measured on the 31-query evaluation set against the 6-document corpus (2,091 chunks, chunk size 300). Raw output: `eval_results/full_results.json`.

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Retrieval Recall@3 | **0.923** | ≥ 0.85 | ✅ |
| Retrieval MRR | **0.817** (dense) / 0.741 (hybrid+reranker) | ≥ 0.70 | ✅ |
| Ragas faithfulness | **0.928** | ≥ 0.85 | ✅ |
| Ragas answer relevancy | **0.960** | ≥ 0.85 | ✅ |
| Refusal accuracy | **1.000** (5/5 out-of-corpus refused) | ≥ 0.80 | ✅ |
| Median end-to-end latency | **~2.3 s** | ≤ 3 s | ✅ |
| Cost per 100 queries | **~$0.02** | — | gpt-4o-mini + text-embedding-3-small |

### Retrieval ablation (k=3, chunk size 300)

| Configuration | Recall@3 | MRR | Avg latency/query |
|---------------|----------|-----|-------------------|
| Dense-only | 0.923 | **0.817** | 0.25 s |
| BM25-only | 0.731 | 0.580 | 0.01 s |
| Hybrid RRF | 0.885 | 0.774 | 0.23 s |
| Hybrid RRF + cross-encoder reranker | **0.923** | 0.741 | 0.74 s |

### Chunk-size ablation (hybrid RRF, no reranker)

| Chunk size | Recall@3 | MRR |
|------------|----------|-----|
| **300** | **0.885** | 0.774 |
| 500 | 0.846 | 0.754 |
| 800 | 0.808 | 0.785 |

### Honest findings

Two results went against the "more machinery is better" assumption, and both are worth more in an interview than a clean sweep would be:

1. **Dense-only retrieval matched the full hybrid+reranker pipeline on Recall@3 (0.923) with the best MRR (0.817) at a third of the latency.** The corpus's natural-language educational prose plays to embedding strengths; BM25's exact-term advantage matters less when eval queries are phrased the way real users ask questions rather than as bare spec identifiers.
2. **The cross-encoder reranker did not improve top-3 quality** — it matched hybrid recall but shuffled MRR downward (0.774 → 0.741). On this corpus the reranker adds latency and memory cost without measurable benefit, which is exactly why ablations belong in production RAG work.

The deployed default keeps hybrid+reranker (configurable per request via the API), but for a memory-constrained deployment, dense-only is the measured-equivalent cheap option.

## Cost Notes

Default OpenAI path:

- Embedding ingestion uses `text-embedding-3-small`.
- Answer generation uses GPT-4o-mini.
- Per-query cost is estimated from observed prompt and completion token counts.

Local path:

- Set `EMBEDDING_PROVIDER=local` for MiniLM embeddings.
- Set `LLM_PROVIDER=ollama` and run Ollama locally for zero API-cost generation.

## Known Limitations

- Scanned PDFs require OCR, which is currently out of scope.
- The default BM25 tokenizer is simple whitespace splitting; measured BM25-only performance (Recall@3 0.731) partly reflects this — technical punctuation needs stronger normalization before BM25 can pull its weight.
- The Federal Register document's 3-column layout extracts imperfectly; the table-lookup eval queries against it are the hardest retrieval cases in the set.
- Streaming responses currently estimate completion tokens unless provider usage metadata is available.
- Render free tier memory may be tight with the cross-encoder reranker loaded — the ablation shows dense-only retrieval is a measured-equivalent fallback.
- Ragas judging jobs can fail transiently (HTTP 431); metrics average over valid samples and report coverage.

*Built by [Alvin Alias](https://github.com/aalias01) · MS Data Science, University of Washington · 12 years HVAC/subsea/manufacturing engineering*
