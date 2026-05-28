# Portfolio Launch Steps

This file separates what has already been handled in the repository from what still needs your choices, API keys, external accounts, or final evaluation runs.

## What Codex Already Completed

- Rewrote `README.md` so it is accurate for the current project state.
- Removed premature claims about final metrics, deployed URLs, and completed ablation results.
- Added a reviewer-friendly status table, setup guide, API examples, evaluation plan, deployment notes, limitations, and interview framing.
- Updated `PROJECT_BRIEF.md` so the talking points describe the work honestly before final metrics exist.
- Converted `data/eval/test_queries.jsonl` into valid JSONL.
- Added `data/eval/README.md` with the required test-set format.
- Validated Python syntax for `api/`, `src/`, and `scripts/`.
- Validated that `src.eval.load_test_queries()` can load the current evaluation scaffold.

## What Codex Can Still Do Without Your External Accounts

These are repo-local tasks I can do once you ask:

1. Add a license file.
   Recommended: MIT if you want the project to be easy for recruiters and reviewers to inspect and reuse.

2. Add a `docs/` folder with portfolio artifacts.
   Useful files:
   - `docs/corpus_selection.md`
   - `docs/evaluation_protocol.md`
   - `docs/deployment_notes.md`
   - `docs/interview_walkthrough.md`

3. Improve the frontend deploy configuration.
   I can make `API_BASE` easier to swap by documenting or implementing a small config pattern.

4. Improve CORS configuration.
   I can move the deployed frontend origin into an environment variable instead of hardcoding the future Vercel URL in `api/main.py`.

5. Add stronger local validation scripts.
   For example:
   - validate JSONL schema
   - check that every expected source document exists in `data/documents/`
   - check that expected source pages are positive integers

6. Add a portfolio-focused pull request summary.
   This can become the GitHub PR description or a release note after final evaluation.

7. Polish screenshots/demo instructions.
   I can add a repeatable script or written process for capturing the UI after the app is running.

## What Requires Your Input Or Accounts

These steps require decisions, files, credentials, or web accounts that I cannot safely invent.

1. Choose the final document corpus.
   You need 5-10 public, license-compatible PDFs that are appropriate to show in a portfolio.

2. Add PDFs to `data/documents/`.
   This folder is gitignored. Do not commit copyrighted standards or private documents.

3. Set your local `.env`.
   Copy `.env.example` to `.env` and add your real API key.

4. Run ingestion.
   This creates the local ChromaDB vector store. The generated `chroma_db/` folder should stay uncommitted.

5. Replace the evaluation examples.
   The current `data/eval/test_queries.jsonl` rows are only examples. The final version should contain about 30 rows tied to your actual PDF corpus.

6. Run evaluation.
   The README results table should only be filled after the retrieval and generation metrics are measured.

7. Deploy backend and frontend.
   Render and Vercel require your accounts and secrets.

8. Add final demo links.
   Update README after deployment:
   - Live demo URL
   - API docs URL

9. Add demo screenshots or a GIF.
   These should show the actual final corpus answering real questions with citations.

## Recommended Final Corpus Strategy

Pick documents that are public, credible, and easy to explain in an interview. Quality matters more than volume.

Suggested mix:

1. OSHA regulations or guidance pages exported/saved as PDFs.
   Good for safety and compliance questions.

2. NASA Technical Reports Server PDFs.
   Good for public engineering reports and technical depth.

3. Public energy-efficiency guides or ASHRAE-adjacent summaries.
   Use publicly accessible guides or excerpts, not restricted paid standards.

4. Public mechanical or piping design educational references.
   Use license-compatible educational material, public agency docs, or manufacturer guides.

Avoid:

- Paid standards copied into the repository
- Private employer documents
- Documents where licensing is unclear
- Scanned PDFs unless you plan to add OCR

## Step-By-Step Launch Plan

### Step 1: Finalize The Corpus

Create a short list before downloading anything:

| Candidate document | Domain | Public source URL | License/access note | Why it belongs |
|--------------------|--------|-------------------|---------------------|----------------|
| Example NASA report | Aerospace/reliability | URL | Public | Tests technical report retrieval |
| Example OSHA document | Safety | URL | Public | Tests regulation retrieval |
| Example energy guide | HVAC/energy | URL | Public | Tests standards-style table lookup |

Decision rule:

- Keep only documents you can confidently discuss in an interview.
- Prefer PDFs with clear page numbers and extractable text.
- Use 5-10 documents total.

### Step 2: Add Documents Locally

Put the final PDFs here:

```bash
mkdir -p data/documents
```

Then manually copy the PDFs into:

```text
data/documents/
```

Check that the folder contains only files you intend to ingest:

```bash
ls data/documents
```

### Step 3: Configure Environment

Create your local environment file:

```bash
cp .env.example .env
```

Edit `.env`:

```text
OPENAI_API_KEY=your-real-key
LLM_PROVIDER=openai
CHROMA_PERSIST_PATH=./chroma_db
```

Optional local model path:

```text
EMBEDDING_PROVIDER=local
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
```

### Step 4: Build The Vector Store

Run ingestion from the project root:

```bash
python -m src.ingestion --reset
```

Expected result:

- PDFs are parsed page by page.
- Chunks are embedded.
- ChromaDB is written to `chroma_db/`.
- The terminal prints total files processed and collection size.

If ingestion says no PDFs were found, check:

```bash
ls data/documents
```

### Step 5: Smoke Test The Pipeline

Run:

```bash
python scripts/smoke_test.py
```

Expected result:

- ChromaDB loads.
- Retriever and generator initialize.
- Sample answers include citations.

If answers are weak, inspect:

- Did the question match the actual corpus?
- Are the PDFs text-extractable?
- Are retrieved chunks from the right document and page?

### Step 6: Build The Final Evaluation Set

Replace the current example rows in:

```text
data/eval/test_queries.jsonl
```

Target:

- 20 in-corpus questions
- 5 borderline questions
- 5 out-of-corpus questions

Each row must be valid JSON on a single line:

```json
{"query": "What requirement does the document give for X?", "expected_source_doc": "exact_filename.pdf", "expected_source_pages": [12], "expected_answer_keywords": ["required term", "numeric value"], "query_type": "in_corpus", "notes": "Tests table lookup from the HVAC guide."}
```

Important:

- `expected_source_doc` must exactly match the PDF filename.
- `expected_source_pages` must match the extracted PDF page numbering used by ingestion.
- Do not add comments to JSONL.

### Step 7: Run Retrieval Evaluation

Run:

```bash
python -m src.eval --retrieval --k 3
```

Record:

- Recall@3
- MRR
- Failed queries
- Whether misses are caused by bad labels, poor chunking, or poor retrieval

Targets:

| Metric | Target |
|--------|--------|
| Recall@3 | >= 0.85 |
| MRR | >= 0.70 |

### Step 8: Run Generation Evaluation

Run:

```bash
python -m src.eval --ragas
```

Record:

- Ragas faithfulness
- Ragas answer relevancy
- Refusal accuracy

Targets:

| Metric | Target |
|--------|--------|
| Faithfulness | >= 0.85 |
| Answer relevancy | >= 0.85 |
| Refusal accuracy | >= 0.80 |

### Step 9: Run Ablations

Use the notebooks to compare:

1. Dense-only retrieval
2. BM25-only retrieval
3. Hybrid dense + BM25 retrieval
4. Hybrid retrieval with and without reranker
5. Chunk sizes around 300, 500, and 800 approximate tokens

Record the winning configuration in the README.

Suggested final table:

| Configuration | Recall@3 | MRR | Notes |
|---------------|----------|-----|-------|
| Dense-only | value | value | Baseline |
| BM25-only | value | value | Exact-term retrieval |
| Hybrid RRF | value | value | Final candidate |
| Hybrid RRF + reranker | value | value | Final setting if latency is acceptable |

### Step 10: Run The App Locally

Start the API:

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

Then open:

```text
frontend/index.html
```

Ask 3-5 final demo questions and confirm:

- The answer is grounded.
- Sources appear.
- Retrieved chunks are inspectable.
- Latency and cost appear.
- Out-of-corpus questions refuse cleanly.

### Step 11: Capture Portfolio Media

Create screenshots or a short GIF showing:

1. Main chat UI with a real engineering question.
2. Answer with citations.
3. Sources panel.
4. Retrieved chunks panel.
5. API docs page if useful.

Save media under:

```text
figures/
```

Recommended filenames:

```text
figures/demo_chat.png
figures/source_citations.png
figures/retrieved_chunks.png
```

### Step 12: Update README With Final Results

Update:

- Live demo link
- API docs link
- Example questions
- Results table
- Cost estimate
- Any known limitations discovered during evaluation

Do not claim:

- A metric that was not measured
- Support for documents not in the corpus
- Production security features that are not implemented

### Step 13: Deploy Backend To Render

On Render:

1. Create a new Blueprint.
2. Connect the GitHub repo.
3. Confirm it reads `render.yaml`.
4. Add secret environment variable:

```text
OPENAI_API_KEY=your-real-key
```

5. Deploy.
6. Check:

```text
https://your-render-service.onrender.com/health
https://your-render-service.onrender.com/docs
```

Important deployment note:

- The deployed service needs access to an ingested vector store or an ingestion/deployment strategy. If `chroma_db/` is not committed, Render will start in degraded mode unless ingestion is run during deployment or the vector store is provisioned another way.

### Step 14: Deploy Frontend To Vercel

On Vercel:

1. Create a new project from the repo.
2. Set project root to:

```text
frontend/
```

3. Update `API_BASE` in `frontend/app.js` to your Render URL.
4. Deploy.

### Step 15: Update Backend CORS

In `api/main.py`, replace:

```text
https://your-project.vercel.app
```

with your real Vercel URL.

Redeploy Render afterward.

### Step 16: Final GitHub Review

Before sharing the repo:

```bash
git status --short
```

Make sure these are not committed:

- `.env`
- `data/documents/`
- `chroma_db/`
- paid/private PDFs
- local caches

Recommended final visible artifacts:

- `README.md`
- `PROJECT_BRIEF.md`
- `PORTFOLIO_LAUNCH_STEPS.md`
- `data/eval/test_queries.jsonl`
- evaluation notebooks
- screenshots/GIFs in `figures/`

### Step 17: Resume Bullet

After metrics are real, use this structure:

```text
Built an end-to-end RAG assistant for engineering documents using ChromaDB, hybrid dense/BM25 retrieval, cross-encoder reranking, GPT-4o-mini, FastAPI, and a streaming JS frontend; achieved Recall@3 of X.XX, MRR of X.XX, and Ragas faithfulness of X.XX on a 30-query evaluation set.
```

Only replace `X.XX` after evaluation is complete.

## Current Blockers

| Blocker | Who Owns It | Why |
|---------|-------------|-----|
| Final PDF corpus | Alvin | Requires source/license judgment |
| OpenAI API key | Alvin | Secret credential |
| Ingested vector store | Alvin/Codex after PDFs and key exist | Requires final documents and embeddings |
| Final evaluation labels | Alvin with Codex help | Requires reading the selected PDFs |
| Hosted Render/Vercel URLs | Alvin | Requires account access |
| Final README metrics | Codex after evaluation runs | Must be based on measured values |

## Best Next Move

Choose the final 5-10 PDFs first. Everything else depends on the corpus: ingestion, evaluation questions, demo prompts, screenshots, metrics, and the final portfolio story.
