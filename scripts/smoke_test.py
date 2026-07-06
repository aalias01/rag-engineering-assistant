"""
scripts/smoke_test.py — Quick sanity check: retrieval + generation on 3 sample queries.

Run this after ingesting documents to verify the full pipeline works end-to-end
before starting the evaluation notebooks.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --query "How does NASA define systems engineering?"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from project root or scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


DEFAULT_QUERIES = [
    "What is the difference between a relief valve and a safety valve?",
    "How often must a process hazard analysis be updated and revalidated under the PSM standard?",
    "How does NASA define systems engineering?",
]


def run_smoke_test(queries: list[str]) -> None:
    print("=" * 70)
    print("RAG Engineering Assistant — Smoke Test")
    print("=" * 70)

    # Check ChromaDB
    try:
        import chromadb
        from src.ingestion import CHROMA_PERSIST_PATH, COLLECTION_NAME
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_PATH))
        collection = client.get_collection(COLLECTION_NAME)
        n_chunks = collection.count()
        print(f"ChromaDB: OK — {n_chunks} chunks in collection '{COLLECTION_NAME}'")
        if n_chunks == 0:
            print("WARNING: Collection is empty. Run `python -m src.ingestion` first.")
            return
    except Exception as e:
        print(f"ChromaDB: FAILED — {e}")
        print("Run `python -m src.ingestion` to ingest documents first.")
        return

    # Check retriever + generator
    try:
        from src.retriever import Retriever
        from src.generator import Generator
        retriever = Retriever(use_hybrid=True, use_reranker=True, top_k=4)
        gen = Generator()
        print("Retriever + Generator: OK")
    except Exception as e:
        print(f"Retriever/Generator init: FAILED — {e}")
        return

    # Run queries
    print()
    for i, query in enumerate(queries, 1):
        print(f"{'─' * 70}")
        print(f"Query {i}: {query}")
        t0 = time.time()
        chunks = retriever.retrieve(query)
        result = gen.generate(query, chunks)
        elapsed = time.time() - t0

        print(f"\nAnswer: {result['answer'][:400]}{'...' if len(result['answer']) > 400 else ''}")
        print("\nSources:")
        for s in result.get("sources", [])[:3]:
            print(f"  {s['source']}  p.{s['page']}")
        print(f"\nStats: chunks={result['chunks_used']}, latency={elapsed:.1f}s, cost=${result.get('cost_usd', 0):.5f}")

    print()
    print("=" * 70)
    print("Smoke test complete. If answers look grounded and citations appear, proceed to notebooks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="Run a single custom query")
    args = parser.parse_args()

    queries = [args.query] if args.query else DEFAULT_QUERIES
    run_smoke_test(queries)
