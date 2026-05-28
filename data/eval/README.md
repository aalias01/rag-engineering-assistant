# Evaluation Test Set

`test_queries.jsonl` must stay valid JSONL: one JSON object per line, no comments and no trailing commas.

Each entry should use this shape:

```json
{"query": "Question text", "expected_source_doc": "filename.pdf", "expected_source_pages": [12], "expected_answer_keywords": ["keyword"], "query_type": "in_corpus", "notes": "Why this query matters"}
```

Use three query types:

- `in_corpus`: answer is clearly present on known page(s)
- `borderline`: answer is present but requires careful retrieval or synthesis across chunks
- `out_of_corpus`: answer is not present and the system should refuse

Before publishing the project, replace the example rows with about 30 final rows tied to the selected PDF corpus.
