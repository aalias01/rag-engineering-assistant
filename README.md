# RAG Engineering Assistant

> Query engineering standards in natural language and receive grounded answers with page-level source citations.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green)](https://www.langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal)](https://fastapi.tiangolo.com/)

RAG Engineering Assistant is an end-to-end GenAI application for technical engineering documents. It ingests PDF standards, manuals, and reports, retrieves relevant excerpts with hybrid search, reranks candidates, and generates answers with page-level citations.

I built this project to show the work behind a real RAG system, not just a chat box on top of PDFs. It covers ingestion, vector search, hybrid retrieval, reranking, grounded prompting, evaluation, API serving, deployment planning, and a custom evidence-review interface.

## Live Demo

- **App (frontend):** `TODO: paste Vercel URL here after deploy`
- **API (backend):** <https://rag-engineering-assistant-api.onrender.com> — see [`/docs`](https://rag-engineering-assistant-api.onrender.com/docs) and [`/health`](https://rag-engineering-assistant-api.onrender.com/health)

The deployed demo runs answer generation on **Groq's free tier** (OpenAI-compatible) and keeps embeddings on OpenAI `text-embedding-3-small`, so it serves live answers at effectively zero cost. The Render free tier sleeps after inactivity, so the first request after idle can take ~30–60 s to wake.

## Highlights

- Built a full RAG application over 778 pages of engineering documents from DOE, NASA, OSHA, and the Federal Register.
- Preserved page-level metadata through ingestion so answers can cite the source document and page.
- Combined dense retrieval, BM25, Reciprocal Rank Fusion, and optional cross-encoder reranking.
- Evaluated retrieval and generation separately on 31 hand-labeled questions.
- Tested refusal behavior with out-of-corpus questions that are close to the engineering domain.
- Exposed the system through FastAPI and a custom frontend for reviewing answers, citations, chunks, latency, and cost.

## Project Snapshot

The system is fully implemented and evaluated on a six-document public-domain engineering corpus with 778 pages and 2,091 chunks. All evaluation targets are met. See [Results](#results).

| Area | Status | Notes |
|------|--------|-------|
| PDF ingestion | Complete | PyMuPDF extraction with pdfplumber fallback, hash-based reingestion skip |
| Corpus | Complete | 6 public-domain documents, 778 pages. See `docs/corpus_selection.md` |
| Vector store | Complete | ChromaDB persistent collection, 2,091 chunks at chunk size 300 (ablation-selected) |
| Retrieval | Complete | Dense retrieval, BM25 retrieval, Reciprocal Rank Fusion, optional cross-encoder reranking |
| Generation | Complete | Groq free tier (deploy default), OpenAI GPT-4o-mini, or local Ollama — selected via `LLM_PROVIDER` |
| API | Complete | FastAPI `GET /health`, `POST /query`, `GET /query/stream` |
| Frontend | Complete | Vanilla JS evidence workbench with streaming, citations, retrieval trace, latency, and cost stats |
| Evaluation set | Complete | 31 hand-labeled queries with verified page-level ground truth |
| Metrics | Complete | Recall@3 0.923, MRR 0.817, faithfulness 0.928, relevancy 0.960, refusal 1.000 |
| Deployment | Ready for hosting | Render (API) + Vercel (frontend) config included; production vector store is committed, so the deployed API ships ready to answer |

## Why I Built It

Engineers spend a lot of time searching standards, procedures, design manuals, and technical reports. Search can find a document, but it usually does not say which section answers the question or whether the answer is grounded in the source.

I built this around the parts of RAG that matter when the system has to be trusted:

- Retrieval quality is measured separately from answer quality.
- Hybrid retrieval handles engineering acronyms and exact specification names better than dense search alone.
- The generator is constrained to answer only from retrieved excerpts.
- Every response includes source document and page citations.
- Out-of-corpus questions are included in the evaluation set.
- The frontend shows the answer, citations, retrieved chunks, latency, and cost in one place.

## The Corpus

Six public-domain engineering documents (778 pages), spanning mechanical fundamentals, systems engineering, and safety/efficiency regulation. Full provenance is in `docs/corpus_selection.md`.

| Document | Pages | Domain |
|----------|-------|--------|
| DOE-HDBK-1012 Vol 1: Thermodynamics | 139 | Mechanical / HVAC fundamentals |
| DOE-HDBK-1012 Vol 2: Heat Transfer | 80 | Mechanical / HVAC fundamentals |
| DOE-HDBK-1018 Vol 2: Valves & Mechanical Components | 130 | Plant equipment |
| NASA Systems Engineering Handbook (SP-2016-6105 Rev 2) | 297 | Systems engineering |
| OSHA 3132: Process Safety Management | 59 | Industrial safety regulation |
| DOE Final Rule 82 FR 1786: Residential CAC/HP Efficiency Standards | 73 | Federal energy regulation |

Every document type here connects to work I have done across 12 years in HVAC, subsea, and manufacturing engineering. The 2017 DOE final rule is the regulation behind the January 2023 product-line redesign I led at Rheem Manufacturing. This corpus is based on a real problem I worked around for years.

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
LLM citation-grounded prompt (Groq / OpenAI / Ollama)
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
| LLM (deploy default) | Groq free tier (`openai/gpt-oss-20b`) | Hosted, OpenAI-compatible, zero-cost generation for the live demo |
| LLM (paid option) | OpenAI GPT-4o-mini | Higher-consistency generation; the model the metrics were measured on |
| Local LLM option | Ollama + Llama 3.1 8B | Zero-cost local/offline generation path |
| API | FastAPI | Health checks, JSON queries, SSE streaming |
| Frontend | HTML, CSS, vanilla JS | Evidence workbench with sources, retrieval trace, and run stats |
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
- Frontend: serve `frontend/` with a static server, then open the local URL

The frontend reads its API base URL from `frontend/config.js` (`window.RAG_CONFIG.API_BASE`). It defaults to `http://localhost:8000`. Change `config.js` or inject `window.RAG_CONFIG` via Vercel settings when pointing at a deployed backend. No edit to `frontend/app.js` is needed.

## API

### `GET /health`

Returns system readiness, ChromaDB load status, collection size, and active LLM provider.

### `POST /query`

Example request:

```json
{
  "query": "How often must a process hazard analysis be updated and revalidated under the PSM standard?",
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

The evaluation set (`data/eval/test_queries.jsonl`) contains 31 hand-labeled queries with verified page-level ground truth. It includes 21 in-corpus questions, 5 borderline questions that require synthesis across chunks, and 5 out-of-corpus questions that should be refused. One hard refusal trap asks about future SEER2/2029 rules using almost the same vocabulary as an in-corpus DOE rule.

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
│   └── deployment_notes.md
├── LICENSE
└── src/
    ├── ingestion.py
    ├── retriever.py
    ├── generator.py
    └── eval.py
```

## Deployment

The live demo runs generation on **Groq's free tier** (OpenAI-compatible, no credit card) and keeps embeddings on OpenAI `text-embedding-3-small`. The production vector store (single `engineering_docs` collection, ~40 MB) is committed to the repo, so the deployed API ships ready to answer — no ingestion step at deploy time.

### Backend on Render

1. Push the repository to GitHub.
2. On Render: **New → Blueprint**, connect this repo. Render reads `render.yaml` (which sets `LLM_PROVIDER=groq`, `GROQ_MODEL`, `EMBEDDING_PROVIDER=openai`, and `USE_RERANKER=false` for the 512 MB free tier).
3. Add two secret environment variables: `GROQ_API_KEY` (free, from [console.groq.com](https://console.groq.com)) and `OPENAI_API_KEY` (used only to embed each incoming query).
4. Deploy, then open `/health` (expect `chroma_loaded: true`, `collection_size: 2091`) and `/docs`.

### Frontend on Vercel

1. Create a Vercel project from the same repo, project root `frontend/`.
2. Point the frontend at the API: set `API_BASE` in `frontend/config.js` to the Render URL (or inject `window.RAG_CONFIG` via Vercel settings).
3. In Render, set `FRONTEND_ORIGIN` to the Vercel URL (CORS is read from env, not hardcoded) and redeploy.

### Switching generation providers

Generation is swappable with one environment variable — no code change:

- `LLM_PROVIDER=groq` — free hosted default (set `GROQ_MODEL`, e.g. `openai/gpt-oss-20b` or the larger `openai/gpt-oss-120b`).
- `LLM_PROVIDER=openai` — GPT-4o-mini (the model the metrics were measured on).
- `LLM_PROVIDER=ollama` — local Llama 3.1 8B for offline/dev.

If you change the *generating* model, re-run `python scripts/run_full_eval.py` before quoting faithfulness/relevancy, since those were measured on GPT-4o-mini.

## Results

All metrics measured on the 31-query evaluation set against the 6-document corpus (2,091 chunks, chunk size 300). Raw output: `eval_results/full_results.json`; the README values come from the `official_results` / `final_config_chunk300` entries.

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Retrieval Recall@3 | **0.923** | ≥ 0.85 | ✅ |
| Retrieval MRR | **0.817** (dense) / 0.741 (hybrid+reranker) | ≥ 0.70 | ✅ |
| Ragas faithfulness | **0.928** | ≥ 0.85 | ✅ |
| Ragas answer relevancy | **0.960** | ≥ 0.85 | ✅ |
| Refusal accuracy | **1.000** (5/5 out-of-corpus refused) | ≥ 0.80 | ✅ |
| Median end-to-end latency | **~2.3 s** | ≤ 3 s | ✅ |
| Cost per 100 queries | **~$0.02** | n/a | gpt-4o-mini + text-embedding-3-small |

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

### What I Learned From The Evaluation

Two findings changed how I would ship this system:

1. **Dense-only retrieval matched the full hybrid+reranker pipeline on Recall@3 (0.923) and had the best MRR (0.817) at about a third of the latency.** The corpus has a lot of natural-language technical prose, so embeddings performed very well.
2. **The cross-encoder reranker did not improve top-3 quality.** It matched hybrid recall, but moved MRR down from 0.774 to 0.741. On this corpus, the reranker adds latency and memory cost without a measurable retrieval gain.

The API still supports hybrid retrieval and reranking per request. For a memory-constrained deployment, dense-only is the simpler and cheaper option based on these results.

## Cost Notes

A RAG query hits a paid API in two places, and they are very unequal: embedding the question costs ~$0.02 per *million* tokens (effectively free), while answer generation is where essentially all the cost lives. The deployment cuts the bill by swapping the generator, not the embedder.

Deployed (free) path:

- Generation runs on **Groq's free tier** (`LLM_PROVIDER=groq`) — $0, no credit card.
- Embeddings stay on OpenAI `text-embedding-3-small`, so each query costs a fraction of a cent; a $5 cap on the OpenAI account is a safe ceiling.

Paid path (what the metrics used):

- `LLM_PROVIDER=openai` → GPT-4o-mini generation, ~$0.02 per 100 queries.

Fully local path:

- `EMBEDDING_PROVIDER=local` (MiniLM) + `LLM_PROVIDER=ollama` (Llama 3.1 8B) for zero external API calls.

## Known Limitations

- Scanned PDFs require OCR, which is currently out of scope.
- The default BM25 tokenizer is simple whitespace splitting. Measured BM25-only performance (Recall@3 0.731) partly reflects this. Technical punctuation needs stronger normalization before BM25 can pull its weight.
- The Federal Register document's 3-column layout extracts imperfectly; the table-lookup eval queries against it are the hardest retrieval cases in the set.
- Streaming responses currently estimate completion tokens unless provider usage metadata is available.
- Render free tier memory may be tight with the cross-encoder reranker loaded. The ablation shows dense-only retrieval is a measured-equivalent fallback.
- Ragas judging jobs can fail transiently (HTTP 431); metrics average over valid samples and report coverage.

*Built by [Alvin Alias](https://github.com/aalias01) | MS Data Science, University of Washington | 12 years HVAC/subsea/manufacturing engineering*
