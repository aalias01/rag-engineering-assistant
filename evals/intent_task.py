"""
evals/intent_task.py — Inspect AI task: intent classification accuracy.

Evaluates a router backend on the stratified holdout split of
data/intent/intent_queries.jsonl. The backend is a task arg, so the same
task definition benchmarks every classifier:

    inspect eval evals/intent_task.py --model mockllm/model -T backend=rules
    inspect eval evals/intent_task.py --model mockllm/model -T backend=zero_shot   # needs GROQ/OPENAI key
    inspect eval evals/intent_task.py --model mockllm/model -T backend=local       # needs trained model

(The classifier calls its own API/model inside the solver; Inspect's --model
is satisfied by mockllm because the solver never calls generate().)

scripts/benchmark_intent.py produces the same numbers plus Wilson CIs and the
README table; this task is the Inspect-native version — .eval logs, seeding,
resume — and the harness the agentic-copilot project will reuse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import match
from inspect_ai.solver import Generate, TaskState, solver

DATA = ROOT / "data" / "intent" / "intent_queries.jsonl"


def _holdout_samples() -> list[Sample]:
    samples = []
    for line in DATA.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != "holdout":
            continue
        samples.append(
            Sample(
                input=row["query"],
                target=row["intent"],
                metadata={"source": row.get("source")},
            )
        )
    return samples


@solver
def intent_solver(backend: str = "rules"):
    from src.router import IntentRouter

    router = IntentRouter(backend=backend)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        decision = router.classify(state.input_text)
        state.output.completion = decision.intent
        state.metadata["method"] = decision.method
        return state

    return solve


@task
def intent_accuracy(backend: str = "rules"):
    return Task(
        dataset=MemoryDataset(_holdout_samples()),
        solver=intent_solver(backend=backend),
        scorer=match(location="exact"),
    )
