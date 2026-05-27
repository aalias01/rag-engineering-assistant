# Project Brief — RAG Engineering Assistant

| Priority Score | Tier | Recommended Ship Slot | Effort |
|----------------|------|----------------------|--------|
| **4.30** | **P1** | **Order #3** *(promoted — Apr 2026 strategic pass; was #5)* | 18–24 hrs across 5 sessions |

**Score breakdown** — ED 5 · DIFF 4 · SC 5 · DSS 4 · BV 4 · EE 3
**Lane:** Modern stack — bridges A (industrial) and B (tech)
**Target companies:** **Microsoft, Amazon (general)**, Boeing, SLB, GE Vernova, Siemens, Honeywell — any company with large technical document libraries; effectively every Fortune 500

**Conditions to re-rank:**
- If applying to a Microsoft / Amazon general DS role: promote — this is the highest tech-credibility signal in the portfolio
- If applying to a research-leaning role (e.g. PNNL, national labs): promote — RAG is the dominant enterprise pattern
- If LLM API costs spike materially: substitute open-source model (Llama 3 / Mistral via Ollama) and document the swap as an additional skill — DON'T deprioritize

**Why promoted from old "Tier 3":** Hiring-manager research consistently surfaces RAG as the most-asked-about modern-stack project. The original brief underweighted Employer Demand. New score 4.30 places it tied for #2 overall.

---

## Problem Statement

Engineers spend 20–30% of their time searching technical documents — standards, manuals, safety codes, design guidelines. The documents exist but are siloed, unsearchable, and require expert context to interpret. A RAG (Retrieval-Augmented Generation) system lets engineers query a corpus of technical PDFs in natural language and get accurate, cited answers grounded in the actual documents — not hallucinated.

**Example queries this system answers:**
- "What is the minimum insulation thickness for a 4-inch pipe carrying 400°F steam per ASHRAE 90.1?"
- "What are the hydrostatic test requirements for Class 900 flanges per ASME B16.5?"
- "What safety factors apply to subsea connector design under API 17D?"

---

## Why This Project for Alvin

- **The highest-signal project for LLM/GenAI skills:** RAG is the dominant enterprise AI pattern in 2025–2026. Every major company is building RAG systems internally. Showing you've built one end-to-end is a major differentiator.
- **Domain gives it credibility:** Generic RAG demos use Wikipedia or random PDFs. Alvin's uses ASHRAE standards, ASME codes, API specs, NASA reports — documents he's actually used in engineering practice. He can validate the outputs.
- **Challenges tech candidates directly:** Pure DS candidates have run notebooks. Alvin ships a working LLM-powered application with vector search, prompt engineering, and citation grounding.
- **Immediately deployable value:** Every engineering company needs this. The demo alone is a conversation piece in any interview.

---

## Dataset — Technical Documents

**Public engineering PDFs (free):**
- ASHRAE 90.1 (energy efficiency standard) — public summaries available
- NASA Technical Reports Server: https://ntrs.nasa.gov (thousands of free technical reports)
- OSHA standards: https://www.osha.gov/laws-regs/regulations/standardnumber (free)
- NFPA 72 (fire protection) — public edition available
- API standards summaries (full standards require purchase — use public excerpts)
- ASME B31.3 (process piping) — public educational materials

**Strategy:** Use 5–10 PDFs from 2–3 domains. Quality of document selection matters more than quantity.

---

## Tech Stack

| Layer | Tool | Justification |
|-------|------|---------------|
| PDF processing | **PyMuPDF** primary, pdfplumber fallback | PyMuPDF preserves layout + tables better |
| Text chunking | LangChain `RecursiveCharacterTextSplitter` | Token-aware; respects natural boundaries |
| Embeddings | **OpenAI text-embedding-3-small** | Cheap (~$0.02 per 1M tokens), fast, strong baseline |
| Optional alternative embeddings | **`all-MiniLM-L6-v2` (sentence-transformers)** | Free, local, demonstrates open-source competence |
| Vector database | **ChromaDB** (local, persisted) | Zero-cost, no cloud setup for dev; production swap to Qdrant/pgvector documented |
| LLM | **OpenAI GPT-4o-mini** | Cost-effective ($0.15 / 1M input, $0.60 / 1M output); capable for grounded Q&A |
| Optional alternative LLM | Llama 3.1 8B via Ollama (local) | Demonstrates open-source path; included as toggle in `.env` |
| RAG orchestration | **LangChain** (LCEL chains) — or LlamaIndex | LangChain's broader ecosystem makes it more recruiter-recognizable |
| Evaluation | **Ragas** + custom test set | Faithfulness, answer relevancy, context precision/recall |
| Observability | LangSmith (optional, free tier) OR JSONL trace logs | Trace every query → retrieved chunks → LLM call → answer |
| API | FastAPI — POST `/query` → answer + sources | Same pattern as CMAPSS, Retail Returns |
| Frontend | Clean chat-style interface (Vercel) — vanilla HTML/CSS/JS | Streaming responses via SSE |
| Environment | conda (`environment.yml`) + pip (`requirements.txt`) | OpenAI key in `.env` (gitignored) |
| Cost control | ~$0.10–0.50 per 100 queries; full corpus ingestion ~$1–3 one-time | Documented in README cost section |
| Version control | git + GitHub repo `aalias01/rag-engineering-assistant` | |

---

## RAG Architecture

```
[PDF Documents]
    ↓ PyMuPDF
[Raw text + metadata (filename, page number)]
    ↓ RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
[Text chunks with metadata]
    ↓ OpenAI text-embedding-3-small
[Dense vectors (1536-dim)]
    ↓ ChromaDB
[Vector store (persisted locally)]
    ↑
[User query] → embed → similarity search → top-k chunks
    ↓
[Retrieved chunks + query] → GPT-4o-mini
    ↓
[Answer with citations: "According to ASHRAE 90.1 Section 6.4.2..."]
```

---

## Key Concepts to Learn and Demonstrate

| Concept | Why it matters |
|---------|---------------|
| Chunking strategy | Too large = noisy retrieval; too small = missing context. Test 300 / 500 / 800 tokens with 10–15% overlap. |
| Embedding models | Maps text to vectors in semantic space. Similar meaning = similar vectors = found by similarity search. |
| Vector similarity search | Cosine similarity or dot product to find most relevant chunks. Not keyword matching. |
| Hybrid retrieval | BM25 keyword + dense embedding combined — important for engineering jargon and acronyms (e.g. ASME, NPSH, COP) that pure semantic search can mis-rank. |
| Reranking | Cross-encoder rerank of top-k retrieved chunks to push the most-relevant chunk to position 1. Big quality lift for ~$0 latency cost. |
| Prompt engineering | Structure the prompt so the LLM answers ONLY from retrieved context. Prevents hallucination. |
| Citation grounding | Every answer includes the source document + page. Hallucination mitigation + trust building. |
| Retrieval evaluation | Does the right chunk get retrieved? Separate from generation quality. Measured with Recall@k and MRR. |
| Faithfulness evaluation | Does the answer match the retrieved context (no fabrication)? Ragas-scored. |
| Cost / latency tradeoffs | Retrieval k, chunk size, rerank, model size all trade quality vs $ vs ms. Documented in the report. |

---

## Prompt Template (Core Engineering)

```python
SYSTEM_PROMPT = """
You are a technical assistant for engineers. Answer questions based ONLY on the
provided document excerpts. If the answer is not in the excerpts, say so clearly.
Do not make up information. Always cite the source document and section number.

Format citations as: [Source: {document_name}, Page {page_number}]
"""

def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join([
        f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
        for c in retrieved_chunks
    ])
    return f"""Context documents:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
```

---

## Evaluation Plan — The Differentiator

Most portfolio RAG demos skip evaluation. Hiring managers know this. Doing it well is the differentiator.

### Test set (built once, used everywhere)

A `data/eval/test_queries.jsonl` file with ~30 entries:

```json
{"query": "What is the minimum pipe insulation thickness for 4-inch steam pipe at 400°F per ASHRAE 90.1?",
 "expected_source_doc": "ASHRAE_90.1_excerpt.pdf",
 "expected_source_pages": [42, 43],
 "expected_answer_keywords": ["1.5 inches", "ASHRAE 90.1", "Section 6.4"]}
```

Mix of: (a) clearly-in-corpus questions, (b) borderline questions answerable with reasoning across chunks, (c) deliberately out-of-corpus questions (system should refuse).

### Metrics

| Metric | What it measures | Acceptable target |
|--------|------------------|-------------------|
| **Retrieval Recall @ 3** | Did the correct chunk appear in top 3? | ≥ 0.85 |
| **Retrieval MRR** | Mean reciprocal rank of correct chunk | ≥ 0.7 |
| **Faithfulness (Ragas)** | Does the answer come from the context? | ≥ 0.85 |
| **Answer Relevancy (Ragas)** | Does the answer address the question? | ≥ 0.85 |
| **Refusal accuracy** | % of out-of-corpus questions correctly refused | ≥ 0.80 |
| **End-to-end latency** | Median time from query to first token | ≤ 3s |
| **Cost per 100 queries** | Total $ for 100 evaluation queries | ≤ $0.50 |

### Ablation experiments

Document at least three side-by-side comparisons in the report:

1. **Chunk size:** 300 vs 500 vs 800 tokens — recall + faithfulness
2. **Retrieval mode:** Dense only vs BM25 only vs Hybrid (dense + BM25)
3. **Reranker on/off:** With and without cross-encoder rerank

This is what separates a demo from a project.

---

## Production Considerations (cite in interview)

Things you've thought about even if not implemented — recruiter-relevant:

- **Document updates / refresh:** Strategy for re-ingesting only changed PDFs (hash-based diff)
- **Access control:** How would you scope retrieval to the user's permission level (mention metadata-filter pattern)
- **Streaming responses:** Already implemented via SSE — good UX, lower perceived latency
- **PII / sensitive data:** Ingestion-time PII detection if expanding to internal docs
- **Cost ceilings per user:** Rate-limiting + monthly cap pattern
- **Graceful degradation:** What happens when the LLM API is down — fall back to top-3 chunks with no synthesis

---

## Deliverables

1. `data/documents/` — curated PDF collection (5–10 engineering documents)
2. `data/eval/test_queries.jsonl` — ~30 evaluation queries with ground truth
3. `notebooks/01_document_processing.ipynb` — PDF extraction, chunking, embedding pipeline
4. `notebooks/02_retrieval_evaluation.ipynb` — Recall@k, MRR, ablation across chunk sizes + retrieval modes
5. `notebooks/03_rag_pipeline.ipynb` — end-to-end RAG with Ragas faithfulness + answer relevancy
6. `notebooks/04_reranker_ablation.ipynb` — cross-encoder rerank on/off comparison
7. `src/ingestion.py` — PDF → chunks → embeddings → ChromaDB
8. `src/retriever.py` — query embedding + similarity search + optional hybrid + rerank
9. `src/generator.py` — prompt construction + GPT-4o-mini call (or local LLM toggle)
10. `src/eval.py` — Ragas + custom retrieval metrics
11. `api/main.py` — FastAPI: POST `/query` → {answer, sources, chunks_used, latency_ms}
12. `frontend/` — clean chat UI: query input → streaming answer + source citations + click-to-expand chunks
13. `README.md` — recruiter-facing with architecture diagram, demo GIF, evaluation table, cost section

---

## Project Phases

### Phase 1 — Document Ingestion Pipeline (3–4 hrs)
- [ ] Select and download 5–10 engineering PDFs (cite sources, license-compatible)
- [ ] Extract text with PyMuPDF, preserve page numbers and document metadata
- [ ] Chunk with RecursiveCharacterTextSplitter (test 300, 500, 800 token chunks)
- [ ] Embed with OpenAI text-embedding-3-small
- [ ] Store in ChromaDB, verify retrieval with quick smoke-test queries

### Phase 2 — Build the Test Set (2 hrs)
- [ ] Hand-author ~30 evaluation queries: in-corpus (20), borderline (5), out-of-corpus (5)
- [ ] Annotate ground truth: source doc, page(s), expected keywords
- [ ] Save as `data/eval/test_queries.jsonl`
- [ ] This is a one-time investment that powers Phases 3–5

### Phase 3 — Retrieval Evaluation + Ablations (3–4 hrs)
- [ ] Implement Recall@k, MRR
- [ ] Run ablation: chunk size 300 / 500 / 800
- [ ] Run ablation: dense only vs BM25 vs Hybrid (e.g. via Reciprocal Rank Fusion)
- [ ] Run ablation: with and without cross-encoder reranker
- [ ] Document the winning configuration with metrics table

### Phase 4 — RAG Pipeline + Faithfulness Evaluation (3–4 hrs)
- [ ] Build prompt template with strict citation requirements
- [ ] GPT-4o-mini integration with retrieved + reranked context
- [ ] Hallucination testing: out-of-corpus queries — measure refusal rate
- [ ] Ragas faithfulness + answer relevancy on the test set
- [ ] Output: full evaluation table in `notebooks/03_rag_pipeline.ipynb`

### Phase 5 — API + Frontend (3–4 hrs)
- [ ] FastAPI: POST `/query` → {answer, sources, chunks_used, latency_ms, cost_estimate_usd}
- [ ] Optional: POST `/ingest` → add new documents to the vector store
- [ ] Streaming response support (SSE)
- [ ] Frontend: chat interface with source citations + click-to-expand retrieved chunks + latency display
- [ ] Deploy Render + Vercel
- [ ] OpenAI API key management (environment variables, not hardcoded)

### Phase 6 — Polish + Cost Documentation (2–3 hrs)
- [ ] Architecture diagram for README
- [ ] Demo GIF or video walkthrough (60–90s)
- [ ] Cost section: per-100 queries, per-1000 queries, full-corpus reingestion
- [ ] Known limitations section: what it gets wrong and why
- [ ] Production-considerations section (see above)
- [ ] Optional: Local-LLM toggle test using Ollama + Llama 3.1 — document quality delta

**Total: 18–24 hours** across ~5 sessions.

---

## Interview Talking Points

1. *"The hardest part of RAG is not the LLM — it's retrieval quality. I evaluated retrieval separately from generation: Recall@3, MRR, and a Ragas faithfulness pass on the final answers. Most demos skip this layer."*
2. *"I tested the system's hallucination resistance by asking out-of-corpus questions explicitly. A well-prompted RAG system should refuse cleanly — mine refuses 80%+ of the time, and the 20% it doesn't are documented as known failure modes."*
3. *"I ran an ablation study — dense-only vs BM25-only vs hybrid. Hybrid won by ~0.08 MRR on engineering documents because acronyms like ASME and NPSH don't always rank well in pure semantic search but are perfect keyword matches."*
4. *"Chunk size matters more than most people realize. 500 tokens with 50-token overlap worked best for multi-page standards because clauses often span paragraph boundaries — but for shorter regulatory snippets 300 was better. I documented both."*
5. *"This is a real production pattern — every major company is building internal RAG systems for their document libraries. I built the full stack: ingestion, retrieval (with reranking), generation, evaluation, API, and frontend. The evaluation table in the README is what an engineering manager would actually want to see before adopting the pattern."*
6. *"The cost story matters in production. I measured ~$0.X per 100 queries with GPT-4o-mini and documented the open-source path (Ollama + Llama 3.1) as a fallback if the cost ceiling drops."*

---

## Success Criteria

- [ ] GitHub repo public; live demo on Vercel + Render
- [ ] System correctly answers ≥ 25/30 test queries from the corpus
- [ ] System correctly refuses ≥ 4/5 out-of-corpus questions
- [ ] Retrieval Recall@3 ≥ 0.85; MRR ≥ 0.7
- [ ] Ragas faithfulness ≥ 0.85
- [ ] At least 3 ablation experiments documented (chunk size, retrieval mode, reranker)
- [ ] Architecture diagram + demo GIF in README
- [ ] Cost section in README
- [ ] Resume bullet: *"Built end-to-end RAG system over engineering standards (ASHRAE, NASA, OSHA); hybrid retrieval (dense + BM25) with cross-encoder rerank, ChromaDB vector store, GPT-4o-mini, FastAPI backend — Recall@3 of X.XX and Ragas faithfulness X.XX on a 30-query evaluation set."*

---

*Brief created: April 2026 · Updated April 2026 (promoted to P1, deeper detail) · May 2026 (ship slot corrected to #3 — promoted from #5 in strategic pass) | Priority Score 4.30 · Tier P1 · Ship slot #3*
