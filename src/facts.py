"""
src/facts.py — Typed facts database for the deterministic lookup path.

The core idea of the v2 routing extension: for factual questions the LLM never
generates the answer. A fact is an atomic, citable value extracted verbatim
from an authoritative document into a JSON-Schema-validated file. Lookup
answers are *templated* from the matched fact — value, unit, citation, and the
verbatim source quote — so a number cannot be hallucinated.

The pattern is domain-agnostic: the engineering corpus here is the demo, but
the same schema/lookup works for compliance thresholds, HR policy limits,
contract terms, or product specs. Swap the facts files; nothing else changes.

Matching is deliberately deterministic (weighted keyword overlap with a
synonym map), not embedding-based: the lookup path exists to be auditable and
exactly evaluable. If matching is uncertain, the router falls back to the
retrieval+LLM interpret path — the facts DB narrows what the LLM does, it
never blocks an answer.

Usage:
    from src.facts import FactsDB
    db = FactsDB()                      # loads data/facts/*.json (validated)
    result = db.lookup("What is the TQ for chlorine?")
    if result.status == "hit":
        print(render_fact_answer(result.fact))
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

FACTS_DIR = Path(os.getenv("FACTS_DIR", "./data/facts"))

# Confidence thresholds for deterministic matching (tuned on the factual eval
# set — see evals/factual_task.py; raise MIN_SCORE to trade coverage for
# precision).
MIN_SCORE = 3.0          # below this: miss → fall through to interpret path
AMBIGUITY_MARGIN = 1.0   # top-2 candidates closer than this → clarify

# Small synonym map applied to query tokens before scoring. Lowercase.
# Domain packs can extend this via a "_synonyms" sidecar later; kept inline
# for now because it is intentionally tiny — heavy lifting belongs to the
# interpret path, not to lookup cleverness.
_SYNONYMS: dict[str, list[str]] = {
    "ac": ["air", "conditioner"],
    "acs": ["air", "conditioner"],
    "a/c": ["air", "conditioner"],
    "hp": ["heat", "pump"],
    "hps": ["heat", "pump"],
    "tq": ["threshold", "quantity"],
    "efficiency": ["seer"],
    "sdhv": ["small", "duct", "high", "velocity"],
    "lbs": ["pounds"],
    "lb": ["pounds"],
    "covered": ["coverage"],
    "cover": ["coverage"],
    "floor": ["minimum"],
    "ceiling": ["maximum"],
    "cap": ["maximum"],
    "interval": ["frequency"],
    "cadence": ["frequency"],
    "schedule": ["frequency"],
    "often": ["frequency"],
}

_STOPWORDS = frozenset(
    """a an and are be but by can could did do does for from has have how in is
    it its me my of on or per required requirement requirements shall should
    tell that the their there this to was were what when where which who
    why will with you your""".split()
)

# Comparator words/symbols normalize to marker tokens so capacity bands
# survive tokenization: "< 45,000 Btu/h" and "under 45,000 Btu/h" both carry
# "lt"; ">= 45,000" and "at least 45,000" both carry "ge".
_LT_WORDS = frozenset(["under", "below", "less", "fewer", "lt"])
_GE_WORDS = frozenset(["least", "greater", "above", "over", "exceeding", "ge"])

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9.,/%-]*")


def _fold_plural(tok: str) -> str:
    """Deterministic plural folding so 'liquids' matches 'liquid' and 'gases'
    matches 'gas'. Crude by design — both sides of every comparison pass
    through the same fold, so consistency matters more than linguistics."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("es") and not tok.endswith("ses"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _token_list(text: str) -> list[str]:
    """Ordered tokens (for bigram matching): lowercase, comparator-normalized,
    hyphen-split, synonym-expanded, plural-folded, stopwords removed."""
    text = (text or "").lower()
    text = text.replace(">=", " ge ").replace("≥", " ge ").replace("<", " lt ").replace(">", " ge ")
    out: list[str] = []
    for raw in _WORD_RE.findall(text):
        # split compounds: "psm-covered" → psm, covered; "split-system" → split, system
        for tok in raw.split("-"):
            tok = tok.strip(".,")
            if not tok or tok in _STOPWORDS:
                continue
            if tok in _LT_WORDS:
                out.append("lt")
                continue
            if tok in _GE_WORDS:
                out.append("ge")
                continue
            if tok in _SYNONYMS:
                out.extend(_fold_plural(t) for t in _SYNONYMS[tok])
            out.append(_fold_plural(tok))
    return out


def _tokenize(text: str) -> set[str]:
    return set(_token_list(text))


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


@dataclass
class Fact:
    fact_id: str
    parameter: str
    entity: str
    value: str
    source_doc: str
    source_page: int
    quote: str
    keywords: list[str]
    unit: str = ""
    qualifier: str = ""
    condition: str = ""
    effective: str = ""
    source_section: str = ""
    notes: str = ""
    domain: str = ""
    curation_status: str = "draft"

    @property
    def display_value(self) -> str:
        return f"{self.value} {self.unit}".strip()


@dataclass
class LookupResult:
    """Outcome of a deterministic lookup.

    status:
        "hit"       — one fact matched with sufficient score and margin
        "ambiguous" — several facts matched; query is missing a qualifier
        "miss"      — nothing matched confidently; use the interpret path
    """

    status: str
    fact: Fact | None = None
    candidates: list[Fact] = field(default_factory=list)
    score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)

    @property
    def clarification(self) -> str:
        """Human-readable disambiguation question for the clarify path."""
        if not self.candidates:
            return ""
        options = sorted({c.qualifier or c.entity for c in self.candidates})
        listing = "; ".join(options[:6])
        return (
            "That question matches more than one value. "
            f"Could you specify which one you mean: {listing}?"
        )


def _validate_facts_file(payload: dict, schema_path: Path) -> None:
    """Validate a facts file against the JSON Schema (fails loudly)."""
    import jsonschema

    schema = json.loads(schema_path.read_text())
    jsonschema.validate(instance=payload, schema=schema)


class FactsDB:
    """Loads and queries all facts files in FACTS_DIR."""

    def __init__(self, facts_dir: Path | str | None = None, validate: bool = True):
        self.facts_dir = Path(facts_dir) if facts_dir else FACTS_DIR
        self.facts: list[Fact] = []
        self._index: list[tuple[Fact, set[str], set[str], set[str]]] = []
        self.load(validate=validate)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, validate: bool = True) -> None:
        self.facts = []
        schema_path = self.facts_dir / "schema.json"
        for path in sorted(self.facts_dir.glob("*.json")):
            if path.name == "schema.json":
                continue
            payload = json.loads(path.read_text())
            if validate and schema_path.exists():
                _validate_facts_file(payload, schema_path)
            curation = payload.get("curation", {})
            for row in payload["facts"]:
                self.facts.append(
                    Fact(
                        fact_id=row["fact_id"],
                        parameter=row["parameter"],
                        entity=row["entity"],
                        qualifier=row.get("qualifier", ""),
                        value=row["value"],
                        unit=row.get("unit", ""),
                        condition=row.get("condition", ""),
                        effective=row.get("effective", ""),
                        source_doc=payload["source_doc"],
                        source_page=row["source_page"],
                        source_section=row.get("source_section", ""),
                        quote=row["quote"],
                        keywords=[k.lower() for k in row["keywords"]],
                        notes=row.get("notes", ""),
                        domain=payload.get("domain", ""),
                        curation_status=curation.get("status", "draft"),
                    )
                )
        self._build_index()

    def _build_index(self) -> None:
        self._index = []
        for f in self.facts:
            param_tokens = _tokenize(f.parameter)
            # Entity coverage uses the CORE name only — parenthetical aliases
            # ("Phosgene (also called Carbonyl Chloride)") would dilute
            # coverage, so they contribute as keywords instead.
            entity_core = f.entity.split("(")[0]
            paren_alias = f.entity[len(entity_core):]
            entity_list = _token_list(entity_core) + _token_list(f.qualifier)
            entity_tokens = set(entity_list)
            entity_bigrams = _bigrams(entity_list)
            # keywords pass through the same normalization as queries so
            # 'pounds' still matches a plural-folded query token; entity and
            # parameter tokens are removed so keywords are purely *additional*
            # evidence, never a double count
            keyword_tokens: set[str] = set()
            for kw in f.keywords:
                keyword_tokens.update(_token_list(kw))
            keyword_tokens.update(_token_list(paren_alias))
            keyword_tokens -= entity_tokens | param_tokens
            self._index.append(
                (f, param_tokens, entity_tokens, entity_bigrams, keyword_tokens)
            )

    def __len__(self) -> int:
        return len(self.facts)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, query: str) -> LookupResult:
        """
        Deterministically match a query to at most one fact.

        Scoring per fact:
            +4.0 × entity coverage (fraction of the fact's entity/qualifier
                  tokens present in the query) — completeness matters, so
                  "chlorine" prefers the Chlorine fact over Chlorine Dioxide
            +1.0 bonus for full entity coverage (the thing is fully named)
            +1.5 bonus if an entity bigram appears in order in the query
                  ("ammonia solutions" beats "ammonia, anhydrous")
            +3.0 × parameter hits × max(parameter coverage, 0.4) — a single
                  generic word ("process") shared with a long parameter name
                  earns little; restating the parameter earns a lot
            +1.0 bonus for parameter coverage ≥ 0.75
            +0.5 per declared keyword present in the query (keywords are
                  de-duplicated against entity/parameter tokens at index time)

        Eligibility gate (prevents cross-parameter matches like a training
        question hitting a SEER2 fact):
            entity coverage ≥ 0.5 ("half-named" the thing), OR
            ≥1 parameter hit with any entity evidence, OR
            ≥2 parameter hits with keyword support or ≥0.5 parameter coverage
        """
        q_token_list = _token_list(query)
        q_tokens = set(q_token_list)
        q_bigrams = _bigrams(q_token_list)
        if not q_tokens:
            return LookupResult(status="miss")

        scored: list[tuple[float, Fact, list[str]]] = []
        for fact, param_toks, entity_toks, entity_bigrams, keyword_toks in self._index:
            param_hits = param_toks & q_tokens
            entity_hits = entity_toks & q_tokens
            keyword_hits = keyword_toks & q_tokens
            entity_cov = len(entity_hits) / len(entity_toks) if entity_toks else 0.0
            param_cov = len(param_hits) / len(param_toks) if param_toks else 0.0
            eligible = (
                entity_cov >= 0.5
                or (len(param_hits) >= 1 and entity_cov > 0)
                or (len(param_hits) >= 2 and (len(keyword_hits) >= 1 or param_cov >= 0.5))
            )
            if not eligible:
                continue
            score = (
                4.0 * entity_cov
                + 3.0 * len(param_hits) * max(param_cov, 0.4)
                + 0.5 * len(keyword_hits)
            )
            if entity_cov == 1.0:
                score += 1.0
            if entity_bigrams & q_bigrams:
                score += 1.5
            if param_cov >= 0.75:
                score += 1.0
            matched = param_hits | entity_hits | keyword_hits
            scored.append((score, fact, matched, entity_cov))

        if not scored:
            return LookupResult(status="miss")

        # Subset dominance (score-independent): a candidate whose matched
        # evidence is a strict subset of another candidate's is not a real
        # alternative — the other fact matched *everything it matched and
        # more* (e.g. the off-mode fact matching only the product-class words
        # of a SEER2 question, or "investigation start" matching two of the
        # three content words of a report-retention question).
        all_matched = [m for _, _, m, _ in scored]
        scored = [
            (score, fact, matched, cov)
            for score, fact, matched, cov in scored
            if not any(matched < other for other in all_matched)
        ]

        scored.sort(key=lambda t: (-t[0], t[1].fact_id))
        best_score, best_fact, best_terms, best_cov = scored[0]

        if best_score < MIN_SCORE:
            return LookupResult(status="miss", score=best_score)

        # Ambiguity: multiple near-tied candidates that differ on a qualifier
        # the query did not pin down (e.g. region, capacity band). A ranking
        # contest is NOT ambiguity: when the winner's entity is (nearly) fully
        # named, weakly-named competitors don't force a clarification.
        near = [
            fact
            for score, fact, _, cov in scored
            if best_score - score < AMBIGUITY_MARGIN
            and not (best_cov >= 0.75 and cov < 0.75)
        ]
        if len(near) > 1:
            distinct_values = {f.display_value for f in near}
            if len(distinct_values) > 1:
                return LookupResult(
                    status="ambiguous",
                    candidates=near[:8],
                    score=best_score,
                    matched_terms=sorted(best_terms),
                )

        return LookupResult(
            status="hit", fact=best_fact, score=best_score, matched_terms=sorted(best_terms)
        )


# ----------------------------------------------------------------------
# Answer rendering (template — no LLM anywhere near this)
# ----------------------------------------------------------------------

def render_fact_answer(fact: Fact) -> str:
    """Render a matched fact as a cited, auditable answer string."""
    qualifier = f" ({fact.qualifier})" if fact.qualifier else ""
    parts = [f"{fact.parameter} for {fact.entity}{qualifier}: **{fact.display_value}**."]
    if fact.condition:
        parts.append(f"Condition: {fact.condition}.")
    if fact.effective:
        parts.append(f"Effective: {fact.effective}.")
    parts.append(f"[Source: {fact.source_doc}, Page {fact.source_page}]")
    parts.append(f'Source text: "{fact.quote}"')
    if fact.curation_status != "verified":
        parts.append(
            "(Draft fact — machine-extracted from the source document and "
            "pending human verification.)"
        )
    return "\n".join(parts)
