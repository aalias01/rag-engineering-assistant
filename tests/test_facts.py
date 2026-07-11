"""Tests for the typed facts DB (v2 lookup path).

Everything here is deterministic and offline — that is the point of the
lookup path, and these tests are the proof.
"""

import json
from pathlib import Path

import pytest

from src.facts import FactsDB, render_fact_answer

ROOT = Path(__file__).resolve().parents[1]
FACTS_DIR = ROOT / "data" / "facts"


@pytest.fixture(scope="module")
def db() -> FactsDB:
    return FactsDB(FACTS_DIR)


def test_facts_files_validate_against_schema(db):
    # FactsDB(validate=True) already ran jsonschema validation on load;
    # loading without error plus a sanity floor is the assertion.
    assert len(db) >= 70


def test_every_fact_has_page_and_quote(db):
    for fact in db.facts:
        assert fact.source_page >= 1, fact.fact_id
        assert len(fact.quote) >= 10, fact.fact_id
        assert fact.source_doc.endswith(".pdf"), fact.fact_id


def test_fact_ids_are_unique(db):
    ids = [f.fact_id for f in db.facts]
    assert len(ids) == len(set(ids))


def test_lookup_hit_named_chemical(db):
    result = db.lookup("What is the threshold quantity for chlorine?")
    assert result.status == "hit"
    assert result.fact.fact_id == "osha_psm_tq_chlorine"
    assert result.fact.value == "1500"


def test_lookup_prefers_complete_entity_over_partial(db):
    # "chlorine" must not be hijacked by Chlorine Dioxide / Trifluoride
    result = db.lookup("TQ for chlorine?")
    assert result.status == "hit"
    assert result.fact.fact_id == "osha_psm_tq_chlorine"


def test_lookup_capacity_band_distinction(db):
    lt = db.lookup(
        "Minimum SEER2 for split-system air conditioners with certified "
        "cooling capacity under 45,000 Btu/h in the Southeast region?"
    )
    ge = db.lookup(
        "Minimum SEER2 for split-system air conditioners with certified "
        "cooling capacity of at least 45,000 Btu/h in the Southeast region?"
    )
    assert lt.status == "hit" and lt.fact.value == "14.3"
    assert ge.status == "hit" and ge.fact.value == "13.8"


def test_lookup_ambiguous_when_values_differ(db):
    # Five hydrogen compounds with different TQs — answering one value
    # would be overconfident, so the lookup reports ambiguity.
    result = db.lookup("What is the TQ for hydrogen?")
    assert result.status == "ambiguous"
    assert len(result.candidates) > 1
    assert "specify" in result.clarification.lower()


def test_lookup_miss_on_conceptual_question(db):
    result = db.lookup("Why is enthalpy referenced to an arbitrary datum?")
    assert result.status == "miss"


def test_lookup_is_deterministic(db):
    q = "How often is refresher training required under PSM?"
    first = db.lookup(q)
    for _ in range(3):
        again = db.lookup(q)
        assert again.status == first.status == "hit"
        assert again.fact.fact_id == first.fact.fact_id


def test_render_includes_citation_quote_and_draft_disclaimer(db):
    result = db.lookup("What is the TQ for phosgene?")
    assert result.status == "hit"
    answer = render_fact_answer(result.fact)
    assert f"Page {result.fact.source_page}" in answer
    assert result.fact.source_doc in answer
    assert "Source text:" in answer
    if result.fact.curation_status != "verified":
        assert "pending human verification" in answer


def test_factual_eval_set_passes_exact_match(db):
    """The frozen factual eval set must pass 100% — it gates the >95% claim.

    This is the CI guard for the deterministic path: if a facts-file or
    matcher change breaks a query, this test names it.
    """
    eval_path = ROOT / "data" / "eval" / "factual_queries.jsonl"
    failures = []
    for line in eval_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result = db.lookup(row["query"])
        if row["expect"] == "clarification":
            ok = result.status == "ambiguous"
        else:
            ok = (
                result.status == "hit"
                and result.fact.value == row["expected_value"]
                and result.fact.source_page == row["expected_page"]
            )
        if not ok:
            failures.append(f"{row['query']!r} -> {result.status}")
    assert not failures, f"{len(failures)} factual eval failures: {failures[:5]}"
