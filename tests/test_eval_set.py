import json
from collections import Counter
from pathlib import Path

from scripts.validate_eval_set import validate_file


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "data" / "eval" / "test_queries.jsonl"


def _load_rows():
    return [
        json.loads(line)
        for line in EVAL_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_eval_set_file_passes_validator_without_doc_folder():
    errors = validate_file(
        eval_path=EVAL_PATH,
        docs_path=ROOT / "data" / "documents",
        skip_doc_check=True,
    )

    assert errors == []


def test_eval_set_has_documented_query_mix_and_page_labels():
    rows = _load_rows()
    counts = Counter(row["query_type"] for row in rows)

    assert len(rows) == 31
    assert counts == {"in_corpus": 21, "borderline": 5, "out_of_corpus": 5}

    for row in rows:
        assert isinstance(row["expected_source_pages"], list)
        assert all(isinstance(page, int) for page in row["expected_source_pages"])
