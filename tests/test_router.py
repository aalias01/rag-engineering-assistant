"""Tests for the intent router (v2) — rules backend and fail-open behavior."""

import pytest

from src.router import VALID_INTENTS, IntentRouter, classify_rules


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is the minimum SEER2 for split-system heat pumps?", "lookup"),
        ("What is the TQ for chlorine?", "lookup"),
        ("How often is refresher training required?", "lookup"),
        ("What is the compliance date for the amended standards?", "lookup"),
        ("Why does the Southwest have a separate EER2 requirement?", "interpret"),
        ("Explain the difference between a relief valve and a safety valve.", "interpret"),
        ("How does a Carnot cycle establish maximum efficiency?", "interpret"),
        ("What is the minimum efficiency?", "clarify"),
        ("", "clarify"),
    ],
)
def test_rules_classifier_canonical_cases(query, expected):
    assert classify_rules(query) == expected


def test_rules_interpret_wins_over_lookup_markers():
    # Reasoning about a lookup-able value is interpret per the labeling policy
    assert classify_rules("Why is the threshold quantity for phosgene so low?") == "interpret"


def test_rules_always_returns_valid_intent():
    for query in ["asdf qwerty", "42", "?" * 50, "the the the"]:
        assert classify_rules(query) in VALID_INTENTS


def test_router_fails_open_to_rules(monkeypatch):
    """A dead zero-shot API must degrade to rules, never raise — the router
    can only add capability over v1, not subtract."""
    monkeypatch.setenv("GROQ_API_KEY", "invalid-key-for-test")
    monkeypatch.setenv("INTENT_PROVIDER", "groq")

    import src.router as router_mod

    def boom(*args, **kwargs):
        raise ConnectionError("network down")

    monkeypatch.setattr(router_mod, "classify_zero_shot", boom)

    router = IntentRouter(backend="zero_shot")
    decision = router.classify("What is the TQ for chlorine?")
    assert decision.intent in VALID_INTENTS
    assert decision.method.startswith("fallback_rules")


def test_router_rules_backend_reports_method():
    router = IntentRouter(backend="rules")
    decision = router.classify("Explain entropy.")
    assert decision.intent == "interpret"
    assert decision.method == "rules"
