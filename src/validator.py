"""
src/validator.py — Post-generation citation validator (v2 extension).

Runs on the interpret path AFTER the LLM answers, BEFORE the answer reaches
the user. Extends the v1 refusal discipline from "the model was told to only
answer from the excerpts" to "we checked":

    1. Citation integrity — every [Source: doc, Page N] citation in the answer
       must point at a chunk that was actually retrieved for this query.
       A citation to a document/page the model never saw is fabricated.
    2. Numeric grounding — every number in the answer must appear in some
       retrieved chunk. Hallucinated numbers are the most dangerous failure
       mode in a technical assistant, and the cheapest to detect.
    3. Sentence support — each content sentence must have sufficient token
       overlap with at least one retrieved chunk (a coarse, deterministic
       faithfulness check; the eval-time RAGAS faithfulness metric is the
       finer instrument, this is the runtime guardrail).

Everything here is deterministic string processing — no LLM, no network —
so it runs identically in tests, CI, and production.

Action policy (VALIDATOR_MODE):
    "flag"   (default) — attach the report; the frontend shows a warning badge
    "strict"           — replace answers that fail citation integrity or
                          numeric grounding with the refusal phrase
    "off"              — skip validation (v1 behavior)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# Matches the citation format the system prompt instructs the model to use:
#   [Source: doe_hdbk_1012_v1_thermodynamics.pdf, Page 80]
# Tolerates minor drift: optional "Pages", multiple pages "80, 81", missing
# ".pdf", varying whitespace.
_CITATION_RE = re.compile(
    r"\[\s*Source:\s*(?P<doc>[^,\]]+?)\s*,\s*Pages?\s*(?P<pages>[\d,\s–-]+)\s*\]",
    re.IGNORECASE,
)

# Numbers worth checking: standalone numerics incl. decimals/thousands
# ("13.4", "10,000", "48", "7"). List markers ("1.", "2)") are excluded by
# _LIST_MARKER_RE below; citation page numbers are stripped before matching.
_NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)(?![\w.])")
_LIST_MARKER_RE = re.compile(r"(?m)^\s*\d+[.)]\s")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9-]+")

_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have if in is it its of on or
    that the this to was were which with must shall should can may not when
    all any each also than then there these those such""".split()
)

MIN_SENTENCE_OVERLAP = 0.35  # fraction of content words found in one chunk
MIN_SUPPORT_COVERAGE = 0.6   # fraction of sentences that must be supported


def _content_words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower())) - _STOPWORDS


def _normalize_doc_name(name: str) -> str:
    name = name.strip().lower()
    if name.endswith(".pdf"):
        name = name[: -len(".pdf")]
    return name


def _parse_pages(pages_str: str) -> list[int]:
    pages: list[int] = []
    for part in re.split(r"[,–-]", pages_str):
        part = part.strip()
        if part.isdigit():
            pages.append(int(part))
    return pages


@dataclass
class ValidationReport:
    status: str = "passed"                       # passed | flagged | refused | not_applicable | off
    citations_found: int = 0
    citations_valid: int = 0
    invalid_citations: list[str] = field(default_factory=list)
    numbers_found: int = 0
    numbers_grounded: int = 0
    ungrounded_numbers: list[str] = field(default_factory=list)
    sentences_checked: int = 0
    sentences_supported: int = 0
    unsupported_sentences: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def support_coverage(self) -> float:
        if self.sentences_checked == 0:
            return 1.0
        return self.sentences_supported / self.sentences_checked

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "citations_found": self.citations_found,
            "citations_valid": self.citations_valid,
            "invalid_citations": self.invalid_citations,
            "numbers_found": self.numbers_found,
            "numbers_grounded": self.numbers_grounded,
            "ungrounded_numbers": self.ungrounded_numbers,
            "sentences_checked": self.sentences_checked,
            "sentences_supported": self.sentences_supported,
            "unsupported_sentences": self.unsupported_sentences[:3],
            "support_coverage": round(self.support_coverage, 3),
            "notes": self.notes,
        }


def validate_answer(
    answer: str,
    chunks: list[dict],
    refusal_checker=None,
) -> ValidationReport:
    """
    Validate a generated answer against the chunks that were retrieved for it.

    `chunks` uses the retriever's dict shape: {"text", "source", "page", ...}.
    `refusal_checker` defaults to src.generator.is_refusal (injected to keep
    this module import-light and independently testable).
    """
    report = ValidationReport()

    if refusal_checker is None:
        from src.generator import is_refusal as refusal_checker  # noqa: PLC0415

    if refusal_checker(answer):
        report.status = "not_applicable"
        report.notes = "Answer is a refusal — nothing to validate."
        return report

    if not chunks:
        report.status = "flagged"
        report.notes = "Non-refusal answer with zero retrieved chunks."
        return report

    chunk_keys = {
        (_normalize_doc_name(c.get("source", "")), int(c.get("page", -1))) for c in chunks
    }
    all_chunk_text = " ".join(c.get("text", "") for c in chunks)
    chunk_numbers = set(_NUMBER_RE.findall(all_chunk_text.replace(",", "")))
    chunk_numbers |= set(_NUMBER_RE.findall(all_chunk_text))
    chunk_word_sets = [_content_words(c.get("text", "")) for c in chunks]

    # --- 1. Citation integrity -------------------------------------------
    for match in _CITATION_RE.finditer(answer):
        report.citations_found += 1
        doc = _normalize_doc_name(match.group("doc"))
        pages = _parse_pages(match.group("pages"))
        ok = any(
            (chunk_doc == doc or doc in chunk_doc or chunk_doc in doc) and page in {p for cd, p in chunk_keys if cd == chunk_doc}
            for chunk_doc, _ in chunk_keys
            for page in pages
        )
        if ok:
            report.citations_valid += 1
        else:
            report.invalid_citations.append(match.group(0))

    # --- 2. Numeric grounding ---------------------------------------------
    answer_body = _CITATION_RE.sub(" ", answer)  # don't test page numbers
    answer_body = _LIST_MARKER_RE.sub(" ", answer_body)  # nor list indices
    for num in _NUMBER_RE.findall(answer_body):
        report.numbers_found += 1
        if num in chunk_numbers or num.replace(",", "") in chunk_numbers:
            report.numbers_grounded += 1
        else:
            report.ungrounded_numbers.append(num)

    # --- 3. Sentence support ----------------------------------------------
    for sentence in _SENTENCE_SPLIT_RE.split(answer_body):
        words = _content_words(sentence)
        if len(words) < 4:  # headers, list markers, stubs
            continue
        report.sentences_checked += 1
        overlap = max(
            (len(words & cw) / len(words) for cw in chunk_word_sets if cw),
            default=0.0,
        )
        if overlap >= MIN_SENTENCE_OVERLAP:
            report.sentences_supported += 1
        else:
            report.unsupported_sentences.append(sentence.strip()[:160])

    # --- Verdict ------------------------------------------------------------
    hard_fail = bool(report.invalid_citations) or bool(report.ungrounded_numbers)
    soft_fail = (
        report.support_coverage < MIN_SUPPORT_COVERAGE
        or (report.citations_found == 0 and report.sentences_checked > 0)
    )
    if hard_fail or soft_fail:
        report.status = "flagged"
    return report


def apply_policy(answer: str, report: ValidationReport, refusal_phrase: str) -> tuple[str, ValidationReport]:
    """Apply VALIDATOR_MODE to a (answer, report) pair.

    In strict mode, hard failures (fabricated citation or ungrounded number)
    replace the answer with a refusal — the same failure contract as v1's
    out-of-corpus refusals, now enforced rather than requested.
    """
    mode = os.getenv("VALIDATOR_MODE", "flag").lower()
    if mode == "off" or report.status in ("passed", "not_applicable"):
        return answer, report
    # Strict-mode refusals require high confidence in the failure: any
    # fabricated citation, or an ungrounded number precise enough (>= 2
    # digits, or a decimal) that a tokenization false positive is unlikely.
    # Single-digit mismatches still flag but never refuse.
    precise_ungrounded = [
        n for n in report.ungrounded_numbers
        if "." in n or len(n.replace(",", "")) >= 2
    ]
    hard_fail = bool(report.invalid_citations) or bool(precise_ungrounded)
    if mode == "strict" and hard_fail:
        report.status = "refused"
        report.notes = (
            "Strict mode: answer withheld — "
            + (
                f"fabricated citation(s): {report.invalid_citations[:2]} "
                if report.invalid_citations
                else ""
            )
            + (
                f"ungrounded number(s): {report.ungrounded_numbers[:3]}"
                if report.ungrounded_numbers
                else ""
            )
        ).strip()
        return refusal_phrase, report
    return answer, report
