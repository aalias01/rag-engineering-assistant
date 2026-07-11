"""Tests for the post-generation citation validator (v2 interpret path)."""

import pytest

from src.generator import REFUSAL_PHRASE
from src.validator import apply_policy, validate_answer

CHUNKS = [
    {
        "text": (
            "Refresher training must be provided at least every three years, "
            "or more often if necessary. The threshold is 10,000 pounds for "
            "flammable liquids and the SEER2 minimum is 13.4."
        ),
        "source": "osha_3132_process_safety_management.pdf",
        "page": 19,
    },
    {
        "text": "Incident investigation reports shall be retained for five years.",
        "source": "osha_3132_process_safety_management.pdf",
        "page": 44,
    },
]


def test_valid_citation_and_grounded_numbers_pass():
    answer = (
        "Refresher training must be provided at least every three years, and "
        "the flammable threshold is 10,000 pounds. "
        "[Source: osha_3132_process_safety_management.pdf, Page 19]"
    )
    report = validate_answer(answer, CHUNKS)
    assert report.status == "passed"
    assert report.citations_valid == report.citations_found == 1
    assert report.ungrounded_numbers == []


def test_fabricated_citation_is_flagged():
    answer = (
        "Training is required every three years. "
        "[Source: nasa_systems_engineering_handbook.pdf, Page 99]"
    )
    report = validate_answer(answer, CHUNKS)
    assert report.status == "flagged"
    assert len(report.invalid_citations) == 1


def test_hallucinated_number_is_flagged_even_single_digit():
    answer = (
        "Refresher training is required every 7 years. "
        "[Source: osha_3132_process_safety_management.pdf, Page 19]"
    )
    report = validate_answer(answer, CHUNKS)
    assert report.status == "flagged"
    assert "7" in report.ungrounded_numbers


def test_list_markers_and_citation_pages_not_treated_as_numbers():
    answer = (
        "1. Provide refresher training at least every three years.\n"
        "2. Keep the 10,000 pounds threshold in mind. "
        "[Source: osha_3132_process_safety_management.pdf, Page 19]"
    )
    report = validate_answer(answer, CHUNKS)
    assert report.status == "passed"
    assert report.ungrounded_numbers == []


def test_refusal_skips_validation():
    report = validate_answer(REFUSAL_PHRASE, CHUNKS)
    assert report.status == "not_applicable"


def test_nonrefusal_with_no_chunks_is_flagged():
    report = validate_answer("Confident answer with no evidence.", [])
    assert report.status == "flagged"


def test_strict_mode_refuses_on_fabricated_citation(monkeypatch):
    monkeypatch.setenv("VALIDATOR_MODE", "strict")
    answer = "The limit is 10,000 pounds. [Source: made_up_doc.pdf, Page 3]"
    report = validate_answer(answer, CHUNKS)
    final_answer, final_report = apply_policy(answer, report, REFUSAL_PHRASE)
    assert final_report.status == "refused"
    assert final_answer == REFUSAL_PHRASE


def test_strict_mode_does_not_refuse_on_single_digit_only(monkeypatch):
    # Single-digit mismatches flag but never refuse (tokenization false
    # positives are too likely at one digit).
    monkeypatch.setenv("VALIDATOR_MODE", "strict")
    answer = (
        "There are 3 core steps to compliance under the standard. "
        "[Source: osha_3132_process_safety_management.pdf, Page 19]"
    )
    report = validate_answer(answer, CHUNKS)
    final_answer, final_report = apply_policy(answer, report, REFUSAL_PHRASE)
    assert final_report.status != "refused"
    assert final_answer == answer


@pytest.mark.parametrize("mode", ["flag", "off"])
def test_non_strict_modes_never_alter_the_answer(monkeypatch, mode):
    monkeypatch.setenv("VALIDATOR_MODE", mode)
    answer = "Bad answer, 99,999 pounds. [Source: made_up.pdf, Page 1]"
    report = validate_answer(answer, CHUNKS)
    final_answer, _ = apply_policy(answer, report, REFUSAL_PHRASE)
    assert final_answer == answer
