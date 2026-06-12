# Evaluation Protocol

This document describes how the assistant is measured. The protocol is intentionally explicit so results in `README.md` are reproducible.

## Test Set

Location: `data/eval/test_queries.jsonl`

Composition (about 30 rows total):

- ~20 in-corpus questions. Answer must come from a specific document and page that exists in `data/documents/`.
- ~5 borderline questions. Answer requires reasoning across multiple chunks, or sits at the edge of corpus coverage.
- ~5 out-of-corpus questions. The system should refuse cleanly.

Each row is valid JSON on a single line with the schema documented in `data/eval/README.md`.

## Retrieval Metrics

Computed by `python -m src.eval --retrieval --k 3`.

| Metric | Definition | Target |
|--------|------------|--------|
| Recall@3 | Fraction of in-corpus queries where any retrieved chunk's `(source, page)` matches the expected answer location | >= 0.85 |
| MRR | Mean reciprocal rank of the first correct retrieved chunk | >= 0.70 |

A "match" is defined as: retrieved chunk's `source` exactly equals `expected_source_doc`, and its `page` is in `expected_source_pages`.

## Generation Metrics

Computed by `python -m src.eval --ragas`.

| Metric | Definition | Target |
|--------|------------|--------|
| Ragas faithfulness | Fraction of answer claims supported by retrieved context | >= 0.85 |
| Ragas answer relevancy | Cosine similarity between question and answer embedding (Ragas formulation) | >= 0.85 |
| Refusal accuracy | Fraction of out-of-corpus queries the system refuses to answer | >= 0.80 |

## Latency And Cost

Captured during `POST /query` from `latency_ms`, `prompt_tokens`, `completion_tokens`, and `cost_usd`. Targets:

| Metric | Target |
|--------|--------|
| Median end-to-end latency | <= 3 seconds |
| Estimated cost per 100 queries | <= $0.50 |

## Ablations

Each ablation must run on the same fixed test set so deltas are comparable.

1. Chunk size — 300 vs 500 vs 800 tokens (retrieval metrics).
2. Retrieval mode — Dense-only vs BM25-only vs Hybrid RRF (retrieval metrics).
3. Reranker on/off — Hybrid RRF without reranker vs Hybrid RRF + cross-encoder reranker (retrieval + generation metrics + latency).

Record the winning configuration and the deltas in the README results table.

## Failure Analysis

For every missed query, classify the failure into exactly one bucket:

- bad-label — the ground truth was wrong; fix the test set.
- bad-chunking — the chunk that contained the answer was split or lost.
- bad-retrieval — the right chunk exists in the store but was not retrieved.
- bad-generation — the right chunk was retrieved but the model did not use it correctly.

Failure counts by bucket should appear in `notebooks/02_retrieval_evaluation.ipynb` and `notebooks/03_rag_pipeline.ipynb`.

## Reproducibility

Every metric reported in the README must be runnable with the documented commands on the documented corpus. If a metric depends on a non-default flag (e.g. `--use_reranker=False`), include the flag next to the number.
