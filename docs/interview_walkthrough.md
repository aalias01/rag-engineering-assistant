# Interview Walkthrough

A 5-7 minute walkthrough script for presenting the project in interviews. Each section maps to a specific moment in the demo.

## 1. Frame The Problem (30 seconds)

"Engineers spend a large portion of their day searching standards, procedures, and technical reports. Generic search returns documents; it doesn't tell you which section actually answers the question, and it doesn't ground the answer in the source. I built an end-to-end RAG assistant that does both."

## 2. Show The UI (60 seconds)

Open the deployed frontend.

- Point at the readiness badge — `Ready · N chunks` — to ground the conversation in real ingested content.
- Ask one in-corpus question (e.g., an OSHA citation question).
- Show the answer, the source pill, and the retrieved-chunk panel.
- Click a chunk open. "Every answer is constrained to the retrieved excerpts; you can see exactly which text the model used."

## 3. Architecture (90 seconds)

Refer to the architecture diagram in `README.md`.

- "Ingestion uses PyMuPDF with a pdfplumber fallback, page-level metadata is preserved."
- "I embed with `text-embedding-3-small` and persist in ChromaDB."
- "Retrieval is hybrid — dense + BM25 fused with Reciprocal Rank Fusion. BM25 matters because engineering text has exact identifiers that pure semantic search miscompares."
- "A cross-encoder reranker promotes the most relevant chunk to position 1 before generation."
- "The generator is GPT-4o-mini with a prompt that constrains it to the retrieved context and requires source citations."

## 4. Evaluation Story (120 seconds) — The Differentiator

"Most RAG demos skip evaluation. I evaluate retrieval and generation separately."

Walk through the results table:

- Recall@3 and MRR — measured against a hand-labeled `test_queries.jsonl` of ~30 questions.
- Ragas faithfulness and answer relevancy — measured on the same set.
- Refusal accuracy — out-of-corpus questions that the system should refuse.
- Ablations table — Dense vs BM25 vs Hybrid; with vs without reranker; 300 vs 500 vs 800 token chunks.

"The winning configuration is hybrid RRF with the reranker; the ablation table shows the deltas that justified picking it."

## 5. Production Considerations (60 seconds)

Pick two from `docs/deployment_notes.md` and `PROJECT_BRIEF.md`:

- "On document updates, ingestion is hash-based — already-ingested PDFs are skipped."
- "Access control would extend the existing metadata filter with a user-scope tag at query time."
- "Cost is bounded by an OpenAI dashboard cap, and I support a local Ollama path for zero-API-cost experimentation."
- "Streaming is via SSE; the frontend renders tokens as they arrive."

## 6. What I'd Do Next (30 seconds)

- Swap ChromaDB for a hosted vector store (Qdrant Cloud) for production multi-tenant use.
- Add ingestion-time PII detection if expanding to internal docs.
- Build a per-query trace view (LangSmith or custom JSONL) for offline debugging.

## Common Follow-Ups

| Question | Short answer |
|----------|--------------|
| Why ChromaDB and not Pinecone? | Zero-cost local dev; production swap is documented. |
| Why GPT-4o-mini? | Strong grounded-Q&A quality at $0.15/$0.60 per 1M tokens; the local Ollama path proves the system is not locked to OpenAI. |
| How do you prevent hallucination? | Strict prompt + retrieved-only context + refusal evaluation in the test set. |
| Why hybrid retrieval? | Engineering text has exact identifiers (ASME B16.5, NPSH, 1910.217); BM25 rescues those cases. |
| How big is the corpus? | 5-10 public PDFs in 2-3 domains. Quality of selection matters more than volume. |

## Demo Failure Recovery

If the live demo fails mid-walkthrough:

- Switch to the README results table and the architecture diagram.
- Open `notebooks/02_retrieval_evaluation.ipynb` to show the actual ablation outputs.
- Note that all numbers in the README are reproducible from the committed evaluation set.
