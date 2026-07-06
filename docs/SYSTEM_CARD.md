# System card

RAG Engineering Assistant answers questions from six public engineering documents. It retrieves document excerpts, asks a generator to answer only from those excerpts, and returns document and page citations with each answer.

## Corpus

The corpus has 778 pages across six public documents:

| Document | Pages |
|---|---:|
| DOE-HDBK-1012 Vol 1, Thermodynamics | 139 |
| DOE-HDBK-1012 Vol 2, Heat Transfer | 80 |
| DOE-HDBK-1018 Vol 2, Mechanical Components | 130 |
| NASA Systems Engineering Handbook SP-2016-6105 Rev 2 | 297 |
| OSHA 3132, Process Safety Management | 59 |
| DOE Final Rule 82 FR 1786, CAC/HP Efficiency Standards | 73 |

The committed Chroma collection has 2,091 chunks at chunk size 300. That chunk size was chosen by ablation.

## Pipeline

1. Page text is extracted with PyMuPDF, with pdfplumber as a fallback for empty pages.
2. Text is split into overlapping chunks with source and page metadata.
3. Dense retrieval uses OpenAI `text-embedding-3-small` by default.
4. BM25 keyword retrieval runs over the same Chroma-backed text.
5. Reciprocal Rank Fusion combines dense and BM25 rankings when hybrid retrieval is on.
6. The optional cross-encoder reranker can reorder candidates.
7. The generator receives the query plus retrieved excerpts and must answer only from those excerpts.

Every API response includes answer text, cited sources, retrieved chunk previews, latency, token counts, model, provider, cost, and a `refused` flag.

## Evaluation

The evaluation set has 31 labeled queries:

| Type | Count |
|---|---:|
| In-corpus | 21 |
| Borderline | 5 |
| Out-of-corpus traps | 5 |

Retrieval and generation are graded separately.

| Metric | Value |
|---|---:|
| Recall@3 | 0.923 |
| MRR | 0.817 |
| Faithfulness | 0.928 |
| Answer relevancy | 0.960 |
| Refusal accuracy | 5 of 5 traps declined |

The README and `eval_results/full_results.json` hold the published results.

## Refusal policy

The system prompt instructs the model to say exactly:

```text
I don't have information on that in the provided documents.
```

The API sets `refused: true` when the answer contains that phrase. This is a heuristic. A model could decline with different wording, but the published refusal result used this instructed phrase.

## Providers and cost

The live deployment uses Groq for generation and OpenAI for query embeddings. The API also supports OpenAI generation and local Ollama generation through `LLM_PROVIDER`.

The cost printed per run comes from the active provider path. Groq and Ollama return `0.0` for generation cost. OpenAI generation uses the estimator in `src/generator.py`.

## Limits

- It answers only from the six documents listed above.
- It should decline questions outside that shelf.
- The evaluation set is 31 labeled queries, not thousands.
- The free Render tier sleeps between visitors, so the first run after idle can take 30 to 60 seconds.
- The cross-encoder reranker is disabled on the free Render service because memory is tight there.
