"""
evals/factual_task.py — Inspect AI task: deterministic lookup path, exact match.

The v2 target: >95% exact match on the factual eval set. Because the lookup
path is deterministic (no LLM anywhere), this eval runs offline, in CI, and
reproduces exactly — a deliberate contrast with the fuzzy RAGAS faithfulness
score on the interpret path.

Two row types (data/eval/factual_queries.jsonl):
    expect="value"          → CORRECT iff the facts DB returns the expected
                              fact value AND cites the expected page
    expect="clarification"  → CORRECT iff the lookup reports ambiguity
                              (answering a value would be overconfident)

Run (no model calls — mockllm satisfies Inspect's model requirement):
    inspect eval evals/factual_task.py --model mockllm/model

Logs land in ./logs as .eval files (Inspect's structured, resumable format).
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

DATA = ROOT / "data" / "eval" / "factual_queries.jsonl"


def _load_samples() -> list[Sample]:
    samples = []
    for i, line in enumerate(DATA.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        samples.append(
            Sample(
                id=f"factual_{i:03d}",
                input=row["query"],
                target=row.get("expected_value") or "CLARIFICATION",
                metadata={
                    "expect": row["expect"],
                    "fact_id": row.get("fact_id"),
                    "expected_page": row.get("expected_page"),
                },
            )
        )
    return samples


@solver
def facts_lookup_solver():
    """Run the deterministic facts lookup — no LLM is invoked."""
    from src.facts import FactsDB, render_fact_answer

    db = FactsDB(ROOT / "data" / "facts")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        result = db.lookup(state.input_text)
        state.metadata["lookup_status"] = result.status
        if result.status == "hit":
            state.metadata["matched_fact_id"] = result.fact.fact_id
            state.metadata["matched_value"] = result.fact.value
            state.metadata["matched_page"] = result.fact.source_page
            state.output.completion = render_fact_answer(result.fact)
        elif result.status == "ambiguous":
            state.output.completion = result.clarification
        else:
            state.output.completion = "(miss — falls through to interpret path)"
        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def facts_exact_match():
    async def score(state: TaskState, target: Target) -> Score:
        expect = state.metadata.get("expect")
        status = state.metadata.get("lookup_status")

        if expect == "clarification":
            ok = status == "ambiguous"
            return Score(
                value=CORRECT if ok else INCORRECT,
                answer=state.output.completion,
                explanation=f"expected ambiguity, lookup status={status}",
            )

        value_ok = status == "hit" and state.metadata.get("matched_value") == target.text
        page_ok = state.metadata.get("matched_page") == state.metadata.get("expected_page")
        ok = value_ok and page_ok
        return Score(
            value=CORRECT if ok else INCORRECT,
            answer=state.output.completion,
            explanation=(
                f"status={status} value_ok={value_ok} page_ok={page_ok} "
                f"(matched {state.metadata.get('matched_fact_id')})"
            ),
        )

    return score


@task
def factual_lookup():
    return Task(
        dataset=MemoryDataset(_load_samples()),
        solver=facts_lookup_solver(),
        scorer=facts_exact_match(),
    )
