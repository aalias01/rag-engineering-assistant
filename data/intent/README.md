# Intent dataset

`intent_queries.jsonl` holds 168 labeled queries for the v2 intent classifier: 56 lookup, 72 interpret, and 40 clarify. The fixed-seed stratified split has 135 training queries and 33 holdout queries. `scripts/build_intent_dataset.py` builds the file from fact-derived templates, hand-written queries, and the 31 retrieval-eval queries relabeled for intent.

## Labeling policy

1. `lookup`: the answer is a single stored value. A number, date, frequency, or threshold, retrievable without synthesis.
2. `interpret` wins whenever reasoning or explanation is requested, even about a lookup-able value. "Why is the TQ for phosgene so low?" is interpret. Out-of-corpus questions are also interpret: routing happens before retrieval, and refusing is the interpret path's job.
3. `clarify`: a competent expert would ask a question back before answering. "What is the minimum efficiency?" names no product class and no standard.

## Status

Reviewed row by row on July 10, 2026. Five generated labels were corrected: three in-corpus single-value questions now route to `lookup`, and two out-of-corpus standards questions now route to `interpret`.

## Regenerating

```bash
python scripts/build_intent_dataset.py
```

Regenerating reshuffles nothing (fixed seed) but does pick up facts-file changes. The holdout split feeds `scripts/benchmark_intent.py` and `evals/intent_task.py`; the train split feeds `scripts/train_intent_classifier.py`.
