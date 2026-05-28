# RAG Engineering Assistant

> Query engineering standards in natural language and receive grounded answers with page-level source citations.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green)](https://www.langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal)](https://fastapi.tiangolo.com/)

RAG Engineering Assistant is an end-to-end retrieval-augmented generation system for technical engineering documents. It ingests PDF standards, manuals, and reports; retrieves relevant excerpts with hybrid search; reranks the best candidates; and generates concise answers grounded only in the retrieved source text.

The project is designed as a portfolio-quality demonstration of practical GenAI engineering: document ingestion, vector search, hybrid retrieval, reranking, citation-grounded prompting, evaluation, API deployment, and a usable chat interface.

## Current Status

This repository contains the application scaffold and core implementation. It is ready for corpus selection, ingestion, evaluation runs, and deployment.

| Area | Status | Notes |
|------|--------|-------|
| PDF ingestion | Implemented | PyMuPDF extraction with pdfplumber fallback, hash-based reingestion skip |
| Vector store | Implemented | ChromaDB persistent local collection |
| Retrieval | Implemented | Dense retrieval, BM25 retrieval, Reciprocal Rank Fusion, optional cross-encoder reranking |
| Generation | Implemented | GPT-4o-mini by default; Ollama local model path supported |
| API | Implemented | FastAPI `GET /health`, `POST /query`, `GET /query/stream` |
| Frontend | Implemented | Vanilla JS chat UI with streaming, citations, retrieved chunk panel, latency/cost stats |
| Evaluation set | Draft scaffold | `data/eval/test_queries.jsonl` contains examples and must be replaced after final corpus selection |
| Metrics | Pending | Run notebooks after ingesting real documents |
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

## Example Questions

These are representative examples. Replace them with corpus-specific examples after the final PDFs are selected and ingested.

```text
What is the minimum pipe insulation thickness for a 4-inch steam pipe at 400 F per ASHRAE 90.1?

What does OSHA 1910.217 require for mechanical power press guarding?

What NASA technical report discusses turbofan engine health monitoring using sensor fusion?
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
Recursive chunking, default about 500 tokens with 50-token overlap
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

The frontend defaults to `http://localhost:8000` in `frontend/app.js`.

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

## Evaluation Plan

The evaluation workflow is intentionally part of the portfolio story. After selecting and ingesting the final corpus, replace the examples in `data/eval/test_queries.jsonl` with about 30 hand-labeled queries:

- About 20 in-corpus questions with known source pages
- About 5 borderline questions requiring careful retrieval
- About 5 out-of-corpus questions that should be refused

Target metrics:

| Metric | Target | Why it matters |
|--------|--------|----------------|
| Retrieval Recall@3 | >= 0.85 | Correct source appears in top results |
| Retrieval MRR | >= 0.70 | Correct source ranks near the top |
| Ragas faithfulness | >= 0.85 | Answer stays grounded in retrieved context |
| Ragas answer relevancy | >= 0.85 | Answer addresses the question |
| Refusal accuracy | >= 0.80 | System avoids answering unsupported questions |
| Median end-to-end latency | <= 3 seconds | UX remains usable |

Planned ablations:

| Experiment | Comparisons |
|------------|-------------|
| Chunk size | 300 vs 500 vs 800 approximate tokens |
| Retrieval mode | Dense-only vs BM25-only vs hybrid RRF |
| Reranking | Hybrid without reranker vs hybrid with cross-encoder reranker |

## Repository Structure

```text
rag-engineering-assistant/
├── README.md
├── PROJECT_BRIEF.md
├── .env.example
├── environment.yml
├── requirements.txt
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
├── scripts/
│   └── smoke_test.py
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
3. Update `API_BASE` in `frontend/app.js` to the Render API URL.
4. Add the Vercel URL to `allow_origins` in `api/main.py`.
5. Redeploy the backend.

## Portfolio Readiness Checklist

Before making the repository public or linking it from a resume:

- [ ] Select 5-10 public or license-compatible engineering PDFs.
- [ ] Replace placeholder examples in `data/eval/test_queries.jsonl` with real corpus-grounded queries.
- [ ] Run ingestion and commit only code, docs, and evaluation artifacts, not PDFs or ChromaDB.
- [ ] Run retrieval and RAG evaluation notebooks.
- [ ] Fill in the metrics table below with actual values.
- [ ] Add a short demo GIF or screenshots to `figures/`.
- [ ] Deploy backend and frontend.
- [ ] Replace pending live demo/API links at the top of this README.
- [ ] Update example questions so they match the final corpus.
- [ ] Add a license file or remove license references from external listings.

For the detailed step-by-step launch workflow, see `PORTFOLIO_LAUNCH_STEPS.md`.

## Results

Fill this table after evaluation runs:

| Metric | Value | Notes |
|--------|-------|-------|
| Retrieval Recall@3 | pending | Target >= 0.85 |
| Retrieval MRR | pending | Target >= 0.70 |
| Ragas faithfulness | pending | Target >= 0.85 |
| Ragas answer relevancy | pending | Target >= 0.85 |
| Refusal accuracy | pending | Target >= 0.80 |
| Median latency | pending | Target <= 3 seconds |
| Estimated cost per 100 queries | pending | Based on observed token use |

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
- The default BM25 tokenizer is simple whitespace splitting; technical punctuation may need stronger normalization for final tuning.
- Streaming responses currently estimate completion tokens unless provider usage metadata is available.
- Render free tier memory may be tight with the cross-encoder reranker loaded.

## Interview Framing

This project is strongest when presented as an evaluated RAG system, not just a chat demo:

- "I evaluated retrieval quality separately from generation quality using Recall@3 and MRR."
- "I included out-of-corpus questions so the system is tested on refusal behavior."
- "I compared dense-only, BM25-only, and hybrid retrieval because engineering standards rely heavily on exact acronyms and specification identifiers."
- "I exposed retrieved chunks and citations in the UI so users can inspect why an answer was produced."

Built by [Alvin Alias](https://github.com/aalias01).
