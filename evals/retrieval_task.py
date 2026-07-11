"""
evals/retrieval_task.py — Inspect AI port of the v1 retrieval eval (31 queries).

Same protocol as notebooks/02 + scripts/run_full_eval.py (Recall@3 against
hand-labeled source pages), re-expressed as an Inspect task so all three v2
evals live in one substrate with .eval logs.

NETWORK REQUIRED: dense retrieval embeds the query with OpenAI
text-embedding-3-small and reads the committed ChromaDB store. Run locally
with OPENAI_API_KEY set:

    inspect eval evals/retrieval_task.py --model mockllm/model

Scoring: CORRECT iff any of the top-3 retrieved chunks comes from the
expected document AND one of the expected pages (±1 page tolerance, matching
the v1 protocol in docs/evaluation_protocol.md). Out-of-corpus rows are
excluded — refusal behavior is generation-side, owned by the RAGAS suite and
tests/test_refusal.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

DATA = ROOT / "data" / "eval" / "test_queries.jsonl"
TOP_K = 3
PAGE_TOLERANCE = 1


def _samples() -> list[Sample]:
    samples = []
    for line in DATA.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("query_type") != "in_corpus":
            continue
        samples.append(
            Sample(
                input=row["query"],
                target=row["expected_source_doc"],
                metadata={"expected_pages": row["expected_source_pages"]},
            )
        )
    return samples


@solver
def retrieval_solver():
    from src.retriever import Retriever

    retriever = Retriever(use_hybrid=True, use_reranker=False, top_k=TOP_K)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        chunks = retriever.retrieve(state.input_text, top_k=TOP_K)
        state.metadata["retrieved"] = [
            {"source": c.get("source"), "page": c.get("page")} for c in chunks
        ]
        state.output.completion = json.dumps(state.metadata["retrieved"])
        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def recall_at_3():
    async def score(state: TaskState, target: Target) -> Score:
        expected_doc = target.text
        expected_pages = state.metadata.get("expected_pages", [])
        hit = any(
            r["source"] == expected_doc
            and any(abs(int(r["page"]) - int(p)) <= PAGE_TOLERANCE for p in expected_pages)
            for r in state.metadata.get("retrieved", [])
        )
        return Score(
            value=CORRECT if hit else INCORRECT,
            answer=state.output.completion,
            explanation=f"expected {expected_doc} pages {expected_pages}",
        )

    return score


@task
def retrieval_recall():
    return Task(
        dataset=MemoryDataset(_samples()),
        solver=retrieval_solver(),
        scorer=recall_at_3(),
    )
