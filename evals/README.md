# Inspect AI evals

V2 moved the evaluation harness to [Inspect AI](https://inspect.aisi.org.uk/). Runs write structured `.eval` logs (gitignored under `logs/`), seed deterministically, and resume after a crash.

| Task | What it grades | Needs network |
|---|---|---|
| `factual_task.py` | Lookup path, exact match on value and page, 50 queries | No |
| `intent_task.py` | Classifier accuracy on the 34-query holdout | Only for `backend=zero_shot` |
| `retrieval_task.py` | Recall@3 port of the 31-query v1 suite | Yes (OpenAI embeddings) |

## Running

```bash
pip install inspect-ai

inspect eval evals/factual_task.py --model mockllm/model
inspect eval evals/intent_task.py --model mockllm/model -T backend=rules
inspect eval evals/retrieval_task.py --model mockllm/model   # needs OPENAI_API_KEY
```

`mockllm/model` satisfies Inspect's model requirement; the solvers call the system's own code and never call `generate()`. View a log with `inspect view`.

The plain-Python equivalents (`scripts/benchmark_intent.py`, `tests/test_facts.py`) report the same numbers with Wilson CIs. CI runs the pytest versions; Inspect is the substrate for eval work going forward.
