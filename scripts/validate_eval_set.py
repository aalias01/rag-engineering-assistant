"""
scripts/validate_eval_set.py — local validation for data/eval/test_queries.jsonl.

Checks (in order):
  1. JSONL parse — every non-blank line is valid JSON.
  2. Schema — required keys present, types correct, query_type in allowed set.
  3. Page integers — every entry in expected_source_pages is a positive int.
  4. Source-doc existence — for in_corpus and borderline rows, the
     expected_source_doc filename exists in data/documents/.

Usage:
    python scripts/validate_eval_set.py
    python scripts/validate_eval_set.py --eval path/to/test_queries.jsonl
    python scripts/validate_eval_set.py --skip-doc-check    # skip step 4

Exit code: 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_KEYS = {
    "query": str,
    "expected_source_doc": str,
    "expected_source_pages": list,
    "expected_answer_keywords": list,
    "query_type": str,
}
ALLOWED_QUERY_TYPES = {"in_corpus", "borderline", "out_of_corpus"}

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL = REPO_ROOT / "data" / "eval" / "test_queries.jsonl"
DEFAULT_DOCS = REPO_ROOT / "data" / "documents"


def _err(line_no: int, msg: str) -> str:
    return f"  line {line_no}: {msg}"


def validate_file(eval_path: Path, docs_path: Path, skip_doc_check: bool) -> list[str]:
    errors: list[str] = []

    if not eval_path.exists():
        return [f"eval file not found: {eval_path}"]

    rows: list[tuple[int, dict]] = []
    with eval_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(_err(line_no, f"invalid JSON ({e.msg})"))
                continue
            if not isinstance(obj, dict):
                errors.append(_err(line_no, "row is not a JSON object"))
                continue
            rows.append((line_no, obj))

    for line_no, obj in rows:
        # Schema
        for key, expected_type in REQUIRED_KEYS.items():
            if key not in obj:
                errors.append(_err(line_no, f"missing required key '{key}'"))
                continue
            if not isinstance(obj[key], expected_type):
                errors.append(
                    _err(line_no, f"key '{key}' must be {expected_type.__name__}")
                )

        qtype = obj.get("query_type")
        if qtype not in ALLOWED_QUERY_TYPES:
            errors.append(
                _err(line_no, f"query_type '{qtype}' not in {sorted(ALLOWED_QUERY_TYPES)}")
            )

        # Page integers
        pages = obj.get("expected_source_pages")
        if isinstance(pages, list):
            for i, p in enumerate(pages):
                if not isinstance(p, int) or isinstance(p, bool) or p < 1:
                    errors.append(
                        _err(line_no, f"expected_source_pages[{i}] must be positive int, got {p!r}")
                    )

        # Source-doc existence (skip for out_of_corpus rows; their doc is "N/A")
        if not skip_doc_check and qtype in {"in_corpus", "borderline"}:
            src = obj.get("expected_source_doc")
            if isinstance(src, str) and src and src != "N/A":
                doc_path = docs_path / src
                if not doc_path.exists():
                    errors.append(
                        _err(
                            line_no,
                            f"expected_source_doc '{src}' not found in {docs_path}",
                        )
                    )

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=Path, default=DEFAULT_EVAL, help="Path to test_queries.jsonl")
    ap.add_argument("--docs", type=Path, default=DEFAULT_DOCS, help="Path to data/documents/")
    ap.add_argument(
        "--skip-doc-check",
        action="store_true",
        help="Skip the check that expected_source_doc exists on disk.",
    )
    args = ap.parse_args()

    print(f"Validating: {args.eval}")
    if not args.skip_doc_check:
        print(f"Doc folder:  {args.docs}")

    errors = validate_file(args.eval, args.docs, args.skip_doc_check)

    if errors:
        print(f"\nFAILED — {len(errors)} issue(s):")
        for e in errors:
            print(e)
        return 1

    print("\nOK — all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
