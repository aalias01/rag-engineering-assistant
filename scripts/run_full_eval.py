"""
scripts/run_full_eval.py — One-shot evaluation suite for the RAG Engineering Assistant.

Runs, in order:
  1. Ingestion at chunk_size=300 (the ablation-selected setting)
  2. Retrieval ablations on the eval set (in_corpus + borderline queries):
       dense-only / BM25-only / hybrid RRF / hybrid RRF + cross-encoder reranker
  3. Chunk-size ablation: re-ingest at 500 and 800, evaluate hybrid retrieval,
     then re-ingest at 300 so the final vector store is the production one
  4. Ragas generation eval (faithfulness, answer relevancy) + refusal accuracy
  5. Latency + token-cost sampling on 5 representative queries

All results are written to eval_results/full_results.json (gitignored corpus
stays out; results JSON is committed so the README numbers are reproducible).

Usage:
    python scripts/run_full_eval.py            # full suite (~10-20 min, ~$1-2)
    python scripts/run_full_eval.py --skip-chunk-ablation   # faster/cheaper
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval import load_test_queries, recall_at_k, reciprocal_rank  # noqa: E402

RESULTS_DIR = ROOT / "eval_results"
TEST_PATH = ROOT / "data" / "eval" / "test_queries.jsonl"
K = 3


def eval_retrieval_fn(retrieve_fn, queries, k=K):
    """Generic retrieval evaluation over a retrieve(query, top_k) callable."""
    recalls, rrs, per_query = [], [], []
    for q in queries:
        chunks = retrieve_fn(q["query"], top_k=k * 3)
        r = recall_at_k(chunks, q["expected_source_doc"], q["expected_source_pages"], k=k)
        rr = reciprocal_rank(chunks, q["expected_source_doc"], q["expected_source_pages"])
        recalls.append(r)
        rrs.append(rr)
        per_query.append({
            "query": q["query"],
            "recall": r,
            "rr": rr,
            "top": f"{chunks[0]['source']} p.{chunks[0]['page']}" if chunks else "none",
            "expected": f"{q['expected_source_doc']} p.{q['expected_source_pages']}",
        })
    n = len(queries)
    return {"recall_at_3": sum(recalls) / n, "mrr": sum(rrs) / n, "n": n, "per_query": per_query}


def run_retrieval_ablations(queries):
    """Four-way retrieval ablation on the current vector store."""
    from src.retriever import Retriever

    r = Retriever()
    out = {}
    print("\n--- Retrieval ablations (k=3) ---")
    for name, fn in [
        ("dense_only", r.retrieve_dense_only),
        ("bm25_only", r.retrieve_bm25_only),
        ("hybrid_rrf", r.retrieve_hybrid_no_rerank),
        ("hybrid_rrf_reranker", r.retrieve),
    ]:
        t0 = time.time()
        res = eval_retrieval_fn(fn, queries)
        res["avg_latency_s"] = round((time.time() - t0) / len(queries), 3)
        out[name] = res
        print(f"  {name:22s} Recall@3={res['recall_at_3']:.3f}  MRR={res['mrr']:.3f}  "
              f"avg {res['avg_latency_s']}s/q")
    return out


def reingest(chunk_size):
    """Re-ingest the corpus at the given chunk size (resets collection)."""
    import importlib
    import src.ingestion as ing
    importlib.reload(ing)
    print(f"\n--- Re-ingesting corpus at chunk_size={chunk_size} ---")
    ing.ingest(reset=True, chunk_size=chunk_size, overlap=max(20, chunk_size // 10))
    # Reset retriever module caches so the next eval sees the new collection
    import src.retriever as ret
    importlib.reload(ret)


def run_chunk_ablation(queries):
    """Evaluate hybrid retrieval at chunk sizes 500 / 800 (300 runs in main pass)."""
    out = {}
    for size in (500, 800):
        reingest(size)
        from src.retriever import Retriever
        r = Retriever(use_reranker=False)
        res = eval_retrieval_fn(r.retrieve_hybrid_no_rerank, queries)
        out[f"chunk_{size}"] = {k: v for k, v in res.items() if k != "per_query"}
        print(f"  chunk_size={size}: Recall@3={res['recall_at_3']:.3f}  MRR={res['mrr']:.3f}")
    return out


def run_generation_eval():
    """Ragas faithfulness/relevancy + refusal accuracy via src.eval."""
    from src.eval import run_ragas_eval
    print("\n--- Ragas generation evaluation ---")
    return run_ragas_eval(TEST_PATH)


def run_latency_cost_sample():
    """Sample end-to-end latency and token cost on 5 representative queries."""
    from src.retriever import Retriever
    from src.generator import Generator

    sample = [
        "What is the difference between a relief valve and a safety valve?",
        "How does NASA define systems engineering?",
        "How often must a process hazard analysis be revalidated?",
        "What SEER level applies to split-system air conditioners in the Southeast?",
        "What is the log mean temperature difference?",
    ]
    r = Retriever()
    g = Generator()
    rows = []
    for q in sample:
        t0 = time.time()
        chunks = r.retrieve(q)
        result = g.generate(q, chunks)
        rows.append({
            "query": q,
            "latency_s": round(time.time() - t0, 2),
            "prompt_tokens": result.get("prompt_tokens"),
            "completion_tokens": result.get("completion_tokens"),
            "cost_usd": result.get("cost_usd"),
        })
        print(f"  {rows[-1]['latency_s']:>5.2f}s  ${rows[-1]['cost_usd']}  {q[:60]}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-chunk-ablation", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    queries = [q for q in load_test_queries(TEST_PATH)
               if q.get("query_type") in ("in_corpus", "borderline")]
    print(f"Loaded {len(queries)} retrieval-evaluable queries from {TEST_PATH.name}")

    out_path = RESULTS_DIR / "full_results.json"

    # Resume support: reload previous partial results if present
    results = {}
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
            print(f"Resuming — found existing results with stages: {list(results.keys())}")
        except Exception:
            results = {}
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["k"] = K

    def save():
        out_path.write_text(json.dumps(results, indent=2, default=str))

    # 1+2. Production ingestion (chunk 300) + retrieval ablations
    if "retrieval_ablations_chunk300" not in results:
        reingest(300)
        results["retrieval_ablations_chunk300"] = run_retrieval_ablations(queries)
        save()

    # 3. Chunk-size ablation (then restore 300)
    if not args.skip_chunk_ablation and "chunk_size_ablation" not in results:
        results["chunk_size_ablation"] = run_chunk_ablation(queries)
        save()
        reingest(300)
        results["final_store"] = "chunk_300"
        save()

    # 4. Generation eval (Ragas + refusal)
    if "generation_eval" not in results:
        results["generation_eval"] = run_generation_eval()
        save()

    # 5. Latency / cost sample
    if "latency_cost_sample" not in results:
        results["latency_cost_sample"] = run_latency_cost_sample()
        save()

    print(f"\nAll results written to {out_path}")


if __name__ == "__main__":
    main()
