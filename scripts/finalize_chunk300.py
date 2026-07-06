"""
scripts/finalize_chunk300.py — Promote chunk_size=300 to production.

The chunk-size ablation showed chunk 300 beats 500 on Recall@3 (0.885 vs 0.846).
This script re-ingests at 300, re-runs the full 4-way retrieval ablation and the
generation eval on the 300-chunk store, and records everything under
'final_config_chunk300' in eval_results/full_results.json.

Usage:
    python scripts/finalize_chunk300.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_full_eval import (  # noqa: E402
    RESULTS_DIR, TEST_PATH, reingest, run_retrieval_ablations,
    run_generation_eval, run_latency_cost_sample,
)
from src.eval import load_test_queries  # noqa: E402


def main():
    out_path = RESULTS_DIR / "full_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    queries = [q for q in load_test_queries(TEST_PATH)
               if q.get("query_type") in ("in_corpus", "borderline")]

    reingest(300)

    final = {"chunk_size": 300, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    final["retrieval_ablations"] = run_retrieval_ablations(queries)

    def json_save():
        out_path.write_text(json.dumps(results, indent=2, default=str))

    results["final_config_chunk300"] = final
    json_save()

    final["generation_eval"] = run_generation_eval()
    json_save()

    final["latency_cost_sample"] = run_latency_cost_sample()
    results["final_store"] = "chunk_300"
    json_save()

    print(f"\nFinal chunk-300 results written to {out_path}")


if __name__ == "__main__":
    main()
