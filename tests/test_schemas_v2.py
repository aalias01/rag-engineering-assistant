"""Tests for the v2 response schema fields — backward compatibility matters:
pre-v2 payloads must still validate (all new fields default)."""

from api.schemas import QueryResponse, ValidationReport


def _base_payload() -> dict:
    return {
        "query": "What is a relief valve?",
        "answer": "A relief valve opens gradually. [Source: doc.pdf, Page 3]",
        "sources": [{"source": "doc.pdf", "page": 3}],
        "chunks_used": 1,
        "chunks": [],
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "cost_usd": 0.0,
        "latency_ms": 900,
        "model": "m",
        "provider": "groq",
        "refused": False,
    }


def test_v1_payload_still_validates_with_defaults():
    response = QueryResponse(**_base_payload())
    assert response.route == "synthesized"
    assert response.intent is None
    assert response.validation is None
    assert response.clarification is None


def test_factual_lookup_payload():
    response = QueryResponse(
        **_base_payload(),
        route="factual_lookup",
        intent="lookup",
        intent_method="rules",
        fact_id="osha_psm_tq_chlorine",
        fact_status="draft",
    )
    assert response.route == "factual_lookup"
    assert response.fact_status == "draft"


def test_validation_report_round_trip():
    report = ValidationReport(
        status="flagged",
        citations_found=2,
        citations_valid=1,
        invalid_citations=["[Source: fake.pdf, Page 9]"],
        numbers_found=3,
        numbers_grounded=2,
        ungrounded_numbers=["99"],
        sentences_checked=4,
        sentences_supported=3,
        support_coverage=0.75,
    )
    response = QueryResponse(**_base_payload(), validation=report)
    assert response.validation.status == "flagged"
    assert response.validation.ungrounded_numbers == ["99"]
