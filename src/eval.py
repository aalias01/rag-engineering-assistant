"""
src/eval.py — Evaluation runner for RAG Engineering Assistant.

Metrics:
    Retrieval:
        - Recall@k: is the ground-truth source doc in the top-k retrieved chunks?
        - MRR (Mean Reciprocal Rank): reciprocal of the rank of the first correct chunk
    Generation:
        - Ragas Faithfulness: does the answer come from the retrieved context (no hallucination)?
        - Ragas Answer Relevancy: does the answer address the question?
        - Refusal accuracy: % of out-of-corpus questions correctly refused

Usage:
    from src.eval import run_retrieval_eval, run_ragas_eval
    results = run_retrieval_eval("data/eval/test_queries.jsonl")
    ragas_results = run_ragas_eval("data/eval/test_queries.jsonl")

    # Or from CLI:
    python -m src.eval --retrieval
    python -m src.eval --ragas
    python -m src.eval --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


TEST_QUERIES_PATH = Path("data/eval/test_queries.jsonl")


# ---------------------------------------------------------------------------
# Load test set
# ---------------------------------------------------------------------------

def load_test_queries(path: Path = TEST_QUERIES_PATH) -> list[dict]:
    """
    Load test queries from a JSONL file.

    Expected format per line:
    {
        "query": "What is the minimum pipe insulation thickness...",
        "expected_source_doc": "ASHRAE_90_1_excerpt.pdf",
        "expected_source_pages": [42, 43],
        "expected_answer_keywords": ["1.5 inches", "Section 6.4"],
        "query_type": "in_corpus"  // "in_corpus" | "borderline" | "out_of_corpus"
    }
    """
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                queries.append(json.loads(line))
    return queries


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_chunks: list[dict], expected_source: str, expected_pages: list[int], k: int) -> int:
    """Returns 1 if any of the top-k chunks contains the expected source doc + at least one expected page."""
    for chunk in retrieved_chunks[:k]:
        if chunk.get("source") == expected_source:
            if any(p == chunk.get("page") for p in expected_pages):
                return 1
    return 0


def reciprocal_rank(retrieved_chunks: list[dict], expected_source: str, expected_pages: list[int]) -> float:
    """Returns 1/rank of the first relevant chunk, or 0 if not found."""
    for i, chunk in enumerate(retrieved_chunks, start=1):
        if chunk.get("source") == expected_source:
            if any(p == chunk.get("page") for p in expected_pages):
                return 1.0 / i
    return 0.0


def run_retrieval_eval(
    test_path: Path = TEST_QUERIES_PATH,
    top_k: int = 3,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run retrieval evaluation on the test set.

    Computes Recall@k and MRR for "in_corpus" and "borderline" queries only
    (out_of_corpus queries are excluded from retrieval eval — they're evaluated
    separately for refusal accuracy in run_ragas_eval).

    Returns:
        {
            "recall_at_k": float,
            "mrr": float,
            "k": int,
            "n_queries": int,
            "n_in_corpus": int,
            "per_query": list[dict],
        }
    """
    from src.retriever import Retriever

    retriever = Retriever(use_hybrid=use_hybrid, use_reranker=use_reranker, top_k=top_k * 3)
    queries = load_test_queries(test_path)
    in_corpus = [q for q in queries if q.get("query_type") in ("in_corpus", "borderline")]

    if not in_corpus:
        print("No in_corpus/borderline queries found in test set.")
        return {}

    recall_scores = []
    rr_scores = []
    per_query = []

    for q in in_corpus:
        chunks = retriever.retrieve(q["query"], top_k=top_k * 3)
        r_at_k = recall_at_k(chunks, q["expected_source_doc"], q["expected_source_pages"], k=top_k)
        rr = reciprocal_rank(chunks, q["expected_source_doc"], q["expected_source_pages"])

        recall_scores.append(r_at_k)
        rr_scores.append(rr)

        per_query.append({
            "query": q["query"][:80] + "..." if len(q["query"]) > 80 else q["query"],
            "recall_at_k": r_at_k,
            "reciprocal_rank": rr,
            "top_result": f"{chunks[0]['source']} p.{chunks[0]['page']}" if chunks else "none",
            "expected": f"{q['expected_source_doc']} p.{q['expected_source_pages']}",
        })

    results = {
        "recall_at_k": sum(recall_scores) / len(recall_scores),
        "mrr": sum(rr_scores) / len(rr_scores),
        "k": top_k,
        "n_queries": len(queries),
        "n_in_corpus": len(in_corpus),
        "per_query": per_query,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Retrieval Evaluation — top_k={top_k}, hybrid={use_hybrid}, reranker={use_reranker}")
        print(f"{'='*60}")
        print(f"Recall@{top_k}:  {results['recall_at_k']:.3f}  (target ≥ 0.85)")
        print(f"MRR:      {results['mrr']:.3f}  (target ≥ 0.70)")
        print(f"Queries:  {len(in_corpus)}/{len(queries)} (in_corpus + borderline)")
        print(f"\nPer-query results:")
        for r in per_query:
            status = "✓" if r["recall_at_k"] else "✗"
            print(f"  {status}  RR={r['reciprocal_rank']:.2f}  Top: {r['top_result']}")
            print(f"     Q: {r['query']}")
        print()

    return results


# ---------------------------------------------------------------------------
# Ragas generation evaluation
# ---------------------------------------------------------------------------

def run_ragas_eval(
    test_path: Path = TEST_QUERIES_PATH,
    top_k: int = 4,
    verbose: bool = True,
) -> dict:
    """
    Run Ragas evaluation (faithfulness + answer relevancy) on the test set.

    For "out_of_corpus" queries, also measures refusal accuracy:
    a refusal is any answer containing phrases like "don't have information",
    "not in the documents", "not provided", "cannot find", etc.

    Returns:
        {
            "faithfulness": float,
            "answer_relevancy": float,
            "refusal_accuracy": float,
            "n_in_corpus": int,
            "n_out_of_corpus": int,
        }
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from datasets import Dataset
    from src.retriever import Retriever
    from src.generator import Generator

    retriever = Retriever(top_k=top_k)
    gen = Generator()
    queries = load_test_queries(test_path)

    in_corpus = [q for q in queries if q.get("query_type") in ("in_corpus", "borderline")]
    out_of_corpus = [q for q in queries if q.get("query_type") == "out_of_corpus"]

    # Build Ragas dataset for in_corpus queries
    ragas_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for q in in_corpus:
        chunks = retriever.retrieve(q["query"])
        result = gen.generate(q["query"], chunks)
        ragas_data["question"].append(q["query"])
        ragas_data["answer"].append(result["answer"])
        ragas_data["contexts"].append([c["text"] for c in chunks])
        # Ragas ground_truth is an expected answer string; use keywords as proxy
        ragas_data["ground_truth"].append(" ".join(q.get("expected_answer_keywords", [])))

    dataset = Dataset.from_dict(ragas_data)
    ragas_scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

    # Refusal accuracy for out-of-corpus queries
    refusal_keywords = [
        "don't have information", "not in the documents", "not provided",
        "cannot find", "no information", "out of scope", "not covered",
        "not in the provided", "cannot answer",
    ]
    refusals = 0
    for q in out_of_corpus:
        chunks = retriever.retrieve(q["query"])
        result = gen.generate(q["query"], chunks)
        answer_lower = result["answer"].lower()
        if any(kw in answer_lower for kw in refusal_keywords):
            refusals += 1

    refusal_accuracy = refusals / len(out_of_corpus) if out_of_corpus else float("nan")

    def _mean_score(value) -> float:
        """
        Ragas returns a scalar in some versions and a per-sample list in others.
        Individual judging jobs can fail transiently (e.g. API errors) and yield
        NaN — average over valid samples and report coverage instead of NaN-ing
        the whole metric.
        """
        import math
        if not isinstance(value, (list, tuple)):
            return float(value)
        vals = [v for v in value if v is not None and not math.isnan(v)]
        if len(vals) < len(value):
            print(f"  note: {len(value) - len(vals)}/{len(value)} samples failed; mean over valid samples")
        return sum(vals) / len(vals) if vals else float("nan")

    results = {
        "faithfulness": _mean_score(ragas_scores["faithfulness"]),
        "answer_relevancy": _mean_score(ragas_scores["answer_relevancy"]),
        "refusal_accuracy": refusal_accuracy,
        "n_in_corpus": len(in_corpus),
        "n_out_of_corpus": len(out_of_corpus),
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Ragas Generation Evaluation")
        print(f"{'='*60}")
        print(f"Faithfulness:      {results['faithfulness']:.3f}  (target ≥ 0.85)")
        print(f"Answer Relevancy:  {results['answer_relevancy']:.3f}  (target ≥ 0.85)")
        print(f"Refusal Accuracy:  {results['refusal_accuracy']:.3f}  (target ≥ 0.80)")
        print(f"In-corpus queries: {len(in_corpus)}")
        print(f"Out-of-corpus:     {len(out_of_corpus)}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--retrieval", action="store_true", help="Run retrieval eval (Recall@k, MRR)")
    parser.add_argument("--ragas", action="store_true", help="Run Ragas eval (faithfulness, relevancy)")
    parser.add_argument("--all", action="store_true", help="Run both")
    parser.add_argument("--k", type=int, default=3, help="Top-k for retrieval eval")
    args = parser.parse_args()

    if args.retrieval or args.all:
        run_retrieval_eval(top_k=args.k)
    if args.ragas or args.all:
        run_ragas_eval()
    if not any([args.retrieval, args.ragas, args.all]):
        parser.print_help()
