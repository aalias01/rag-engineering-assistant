"""
scripts/benchmark_intent.py — Benchmark intent classifiers on the holdout set.

Compares every available backend on the same fixed holdout split:

    rules      — deterministic keyword baseline (always available, free)
    zero_shot  — Groq (free tier) and/or OpenAI GPT-4o-mini (needs API keys)
    local      — fine-tuned DistilBERT+LoRA (needs models/intent_distilbert/)

"Benchmark honestly, keep the winner" is the v2 brief's instruction: the
router's INTENT_CLASSIFIER env var should be set to whichever row wins here,
and the README table should show all rows — including the baseline the fancy
options must beat.

Reports accuracy with Wilson 95% CIs because the holdout set is small, plus a
per-backend confusion table.

Usage (from repo root):
    python scripts/benchmark_intent.py                    # all available
    python scripts/benchmark_intent.py rules              # subset
    python scripts/benchmark_intent.py rules zero_shot_groq

Outputs:
    eval_results/intent_benchmark.json
    docs/intent_benchmark.md   (markdown table for the README)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "intent" / "intent_queries.jsonl"
RESULTS_JSON = ROOT / "eval_results" / "intent_benchmark.json"
RESULTS_MD = ROOT / "docs" / "intent_benchmark.md"

LABELS = ["clarify", "interpret", "lookup"]


def holdout_rows() -> list[dict]:
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    return [r for r in rows if r["split"] == "holdout"]


def available_backends(requested: list[str]) -> dict[str, callable]:
    """Map backend name → classify(query) callable, for what's usable here."""
    import os

    from src.router import classify_local, classify_rules, classify_zero_shot

    backends: dict[str, callable] = {"rules": classify_rules}

    if os.getenv("GROQ_API_KEY"):
        backends["zero_shot_groq"] = lambda q: classify_zero_shot(q, provider="groq")
    if os.getenv("OPENAI_API_KEY"):
        backends["zero_shot_gpt4o_mini"] = lambda q: classify_zero_shot(q, provider="openai")
    model_dir = ROOT / "models" / "intent_distilbert"
    if model_dir.exists():
        backends["local_distilbert_lora"] = lambda q: classify_local(q, str(model_dir))

    if requested:
        backends = {k: v for k, v in backends.items() if k in requested}
    return backends


def main() -> None:
    from src.stats import format_proportion, wilson_ci

    requested = [a for a in sys.argv[1:] if not a.startswith("-")]
    rows = holdout_rows()
    backends = available_backends(requested)
    print(f"holdout n={len(rows)}  backends={list(backends)}\n")

    results: dict[str, dict] = {}
    for name, classify in backends.items():
        correct = 0
        latencies: list[float] = []
        confusion = {a: {b: 0 for b in LABELS} for a in LABELS}
        errors = 0
        for row in rows:
            t0 = time.time()
            try:
                pred = classify(row["query"])
            except Exception as e:
                errors += 1
                print(f"  [{name}] ERROR on {row['query']!r}: {e}")
                pred = "interpret"  # what the router's fail-open would do
            latencies.append((time.time() - t0) * 1000)
            confusion[row["intent"]][pred] += 1
            correct += int(pred == row["intent"])

        low, high = wilson_ci(correct, len(rows))
        per_class = {
            label: {
                "n": sum(confusion[label].values()),
                "correct": confusion[label][label],
            }
            for label in LABELS
        }
        results[name] = {
            "n": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows),
            "wilson_95": [round(low, 4), round(high, 4)],
            "per_class": per_class,
            "confusion": confusion,
            "median_latency_ms": sorted(latencies)[len(latencies) // 2],
            "api_errors": errors,
        }
        print(f"{name}: {format_proportion(correct, len(rows))}  "
              f"(median {results[name]['median_latency_ms']:.0f} ms, errors {errors})")

    RESULTS_JSON.parent.mkdir(exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2) + "\n")

    lines = [
        "# Intent classifier benchmark",
        "",
        f"Holdout set: {len(rows)} queries (stratified 20% of "
        "`data/intent/intent_queries.jsonl`, seed 558). n is small, so the "
        "Wilson 95% CI is the claim that matters.",
        "",
        "| Backend | Accuracy | 95% CI | Median latency | Notes |",
        "|---|---|---|---|---|",
    ]
    notes = {
        "rules": "keyword baseline, deterministic, $0",
        "zero_shot_groq": "free tier, no training data needed",
        "zero_shot_gpt4o_mini": "paid API",
        "local_distilbert_lora": "LoRA fine-tune, offline, $0/query",
    }
    for name, r in results.items():
        low, high = r["wilson_95"]
        lines.append(
            f"| {name} | {r['accuracy']:.1%} ({r['correct']}/{r['n']}) "
            f"| {low:.1%}–{high:.1%} | {r['median_latency_ms']:.0f} ms "
            f"| {notes.get(name, '')} |"
        )
    lines.append("")
    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {RESULTS_JSON}\nwrote {RESULTS_MD}")


if __name__ == "__main__":
    main()
