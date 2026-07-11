"""Integration tests for predictor routing (v2) — no ChromaDB, no network.

Uses the rules classifier and the real facts DB; the interpret path is not
exercised here (it needs retrieval + an LLM and is covered by the existing
integration suite and the RAGAS eval).
"""

import pytest

from api import predictor


@pytest.fixture(autouse=True)
def rules_router(monkeypatch):
    monkeypatch.setenv("INTENT_CLASSIFIER", "rules")
    monkeypatch.setenv("ROUTER_ENABLED", "true")
    predictor._load_routing()
    yield
    predictor._router = None
    predictor._facts_db = None


def test_route_lookup_hit():
    path, meta, payload = predictor._route("What is the TQ for chlorine?")
    assert path == "lookup"
    assert meta["intent"] == "lookup"
    assert payload.fact_id == "osha_psm_tq_chlorine"


def test_route_interpret_for_conceptual():
    path, meta, _ = predictor._route("Why does entropy increase in real processes?")
    assert path == "interpret"
    assert meta["intent"] == "interpret"


def test_route_ambiguous_lookup_becomes_clarify():
    path, _, clarification = predictor._route("What is the TQ for hydrogen?")
    assert path == "clarify"
    assert "specify" in clarification.lower()


def test_route_lookup_miss_falls_through_to_interpret():
    # Lookup-shaped question about a value the facts DB doesn't store
    path, meta, _ = predictor._route(
        "What is the maximum allowable working pressure for carbon steel pipe?"
    )
    assert path == "interpret"


def test_lookup_response_envelope():
    _, _, fact = predictor._route("What is the TQ for chlorine?")
    result = predictor._lookup_response("q", fact)
    assert result["route"] == "factual_lookup"
    assert result["provider"] == "deterministic"
    assert result["cost_usd"] == 0.0
    assert result["refused"] is False
    assert result["sources"][0]["page"] == fact.source_page
    assert result["chunks"][0]["retrieval_method"] == "facts_db"
    assert "1500" in result["answer"]


def test_clarify_response_envelope():
    result = predictor._clarify_response("q", "Which one do you mean?")
    assert result["route"] == "clarification"
    assert result["clarification"] == "Which one do you mean?"
    assert result["chunks_used"] == 0
    assert result["refused"] is False


def test_router_disabled_env_means_v1_behavior(monkeypatch):
    monkeypatch.setenv("ROUTER_ENABLED", "false")
    path, meta, payload = predictor._route("What is the TQ for chlorine?")
    assert path == "interpret"
    assert meta["intent"] is None
