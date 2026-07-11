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

## V2: routing, facts path, validation

V2 classifies each query before retrieval and routes it down one of three paths.

| Route | Mechanism | Model involvement |
|---|---|---|
| `factual_lookup` | Value assembled from `data/facts/*.json` with quote and page | None |
| `synthesized` | v1 retrieve + generate, then citation validation | Generation only |
| `clarification` | Disambiguation question from the facts lookup | None |

The facts files hold 88 verified facts from the DOE 2017 final rule and OSHA 3132. Each fact stores the source wording and page. `scripts/build_facts_db.py` checks every quote against the cited PDF page and fails the build on a miss. The facts were checked against the cited pages and marked verified on July 10, 2026.

The classifier has three backends (`INTENT_CLASSIFIER`): a zero-shot prompt to Groq or OpenAI, a locally fine-tuned DistilBERT with LoRA, and a keyword baseline. GPT-4o-mini had the best point estimate on the reviewed holdout, but production uses Groq because the two-answer difference is inconclusive on this small set and Groq avoids classifier cost. Any backend failure degrades to the keyword rules, and a lookup miss falls through to the synthesized path.

Production provider calls by route:

| Route | Groq | OpenAI |
|---|---|---|
| `factual_lookup` | Intent classification | None |
| `clarification` | Intent classification | None |
| `synthesized` | Intent classification and answer generation | Query embedding |
| `/health` | None | None |

GPT-4o-mini was benchmarked as an alternative classifier. It is not selected in production. Groq serves `openai/gpt-oss-20b`; the `openai/` prefix is part of the model ID, not the API or billing provider.

The validator runs on the synthesized path after generation. It checks that each `[Source: doc, Page N]` citation names a chunk that was retrieved for this query, that each number in the answer appears in a retrieved chunk, and that each sentence has token overlap with at least one chunk. `VALIDATOR_MODE=flag` attaches the report to the response. `strict` replaces hard failures with the refusal phrase on `POST /query`. The streaming endpoint cannot withdraw tokens already sent, so there the report is advisory and the frontend badge carries it.

V2 measured results:

| Check | Result |
|---|---:|
| Factual exact match (value and page) | 50 of 50 |
| Quote verification | 88 of 88 |
| Intent accuracy, GPT-4o-mini (33-query holdout) | 31 of 33, Wilson 95% CI 80.4% to 98.3% |
| Intent accuracy, Groq (33-query holdout) | 29 of 33, Wilson 95% CI 72.7% to 95.2% |
| Intent accuracy, local DistilBERT + LoRA (33-query holdout) | 27 of 33, Wilson 95% CI 65.6% to 91.4% |
| Intent accuracy, rules baseline (33-query holdout) | 21 of 33, Wilson 95% CI 46.6% to 77.8% |

The small-sample confidence intervals overlap, so the ranking is not conclusive. The full output is in `docs/intent_benchmark.md`.

## Providers and cost

The live deployment uses Groq for intent classification and answer generation. OpenAI embeds interpretation queries before retrieval. Fact lookups and clarifications do not call OpenAI. The API also supports OpenAI generation and local Ollama generation through `LLM_PROVIDER`.

The cost printed per run comes from the active provider path. Groq and Ollama return `0.0` for generation cost. OpenAI generation uses the estimator in `src/generator.py`.

## Limits

- It answers only from the six documents listed above.
- It should decline questions outside that shelf.
- The evaluation set is 31 labeled queries, not thousands. The v2 factual set is 50 and the intent holdout is 33.
- The current facts files were extracted with AI assistance, checked against their cited PDF pages, and approved by Alvin Alias on July 10, 2026. Newly added draft facts remain labeled until they pass the same check.
- Facts lookup is keyword matching, not semantic search. A phrasing it has never seen can miss and fall through to the synthesized path. That is the designed failure direction.
- The validator's sentence-support check is token overlap, which is coarse. The eval-time faithfulness metric remains the finer instrument.
- The free Render tier sleeps between visitors, so the first run after idle can take 30 to 60 seconds.
- The cross-encoder reranker is disabled on the free Render service because memory is tight there.
