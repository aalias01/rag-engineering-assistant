"""
src/router.py — Intent classification and query routing (v2 extension).

Every query is classified into one of three intents before retrieval:

    lookup    — asks for a specific stored value (a number, a threshold, a
                date). Served by the typed facts DB (src/facts.py); the LLM
                never generates the answer.
    interpret — asks for explanation, comparison, reasoning. Served by the
                v1 retrieval + grounded-generation pipeline, now with a
                post-generation citation validator (src/validator.py).
    clarify   — too underspecified to answer responsibly. Served by a
                clarification question instead of a guess.

Three interchangeable classifier backends, selected via INTENT_CLASSIFIER:

    "zero_shot" (default) — few-shot prompt to an OpenAI-compatible chat API
                            (Groq free tier by default, INTENT_PROVIDER=openai
                            for GPT-4o-mini). Temperature 0, JSON output.
    "local"               — fine-tuned DistilBERT + LoRA adapter trained by
                            scripts/train_intent_classifier.py (fully offline).
    "rules"               — deterministic keyword heuristics. Also the honest
                            cheap baseline in the benchmark, and the runtime
                            fallback when an API call fails.

Design rule: FAIL OPEN. Any classifier error routes to "interpret", which is
exactly the v1 behavior — the router can only add capability, never take the
system below its v1 baseline.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

VALID_INTENTS = ("lookup", "interpret", "clarify")

ZERO_SHOT_SYSTEM_PROMPT = """You classify user queries for a document question-answering system.

Choose exactly one intent:
- "lookup": the query asks for a specific stored value — a number, limit,
  threshold, date, frequency, or named specification (e.g. "What is the
  minimum SEER2 for split-system heat pumps?", "TQ for chlorine?",
  "How often is refresher training required?").
- "interpret": the query asks for explanation, definition, comparison,
  reasoning, or procedure (e.g. "Why is enthalpy given with respect to a
  reference value?", "Explain the difference between a relief valve and a
  safety valve.").
- "clarify": the query is too underspecified to answer responsibly (e.g.
  "What's the minimum efficiency?" — of what product class, under which
  standard?).

Respond with JSON only: {"intent": "<lookup|interpret|clarify>"}"""

# ---------------------------------------------------------------------------
# Rules baseline / fallback
# ---------------------------------------------------------------------------

_LOOKUP_PATTERNS = [
    r"\bwhat is the (minimum|maximum|required|threshold|allowed)\b",
    r"\b(minimum|maximum) (seer2?|hspf2?|eer2?|efficiency|value)\b",
    r"\b(seer2?|hspf2?|eer2?)\b.*\b(for|of)\b",
    r"\bthreshold quantity\b",
    r"\btq\b",
    r"\bhow (often|many|much|long)\b",
    r"\bwhat (quantity|amount|value|date|frequency)\b",
    r"\b(effective|compliance) date\b",
    r"\bhow (frequently|regularly)\b",
    r"\bat least every\b",
    r"\boff.?mode\b.*\b(standard|watts?)\b",
]

_CLARIFY_PATTERNS = [
    r"^\s*(what('| i)?s| what is)? ?the (minimum|maximum|limit|requirement)s?\s*\??\s*$",
    r"\b(minimum|maximum) efficiency\s*\?*\s*$",
    r"^\s*what about\b",
    r"^\s*(and|what about|how about) (the )?(other|rest|others)\b",
    r"\b(it|its|they|them|that one)\b.*\?\s*$",
]

_INTERPRET_MARKERS = re.compile(
    r"\b(why|explain|how does|how do|describe|difference|compare|what happens|purpose of|when should)\b",
    re.IGNORECASE,
)


def classify_rules(query: str) -> str:
    """Deterministic keyword classifier — baseline and runtime fallback."""
    q = (query or "").strip().lower()
    if not q:
        return "clarify"
    # Interpret markers win first: "why is the TQ for chlorine so low?" asks
    # for reasoning even though it names a lookup-able value.
    if _INTERPRET_MARKERS.search(q):
        return "interpret"
    for pat in _CLARIFY_PATTERNS:
        if re.search(pat, q):
            return "clarify"
    for pat in _LOOKUP_PATTERNS:
        if re.search(pat, q):
            return "lookup"
    return "interpret"


# ---------------------------------------------------------------------------
# Zero-shot backend (OpenAI-compatible: Groq default, OpenAI optional)
# ---------------------------------------------------------------------------

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_clients: dict = {}


def _get_client(provider: str):
    if provider not in _clients:
        from openai import OpenAI

        if provider == "groq":
            _clients[provider] = OpenAI(
                base_url=_GROQ_BASE_URL, api_key=os.getenv("GROQ_API_KEY")
            )
        else:
            _clients[provider] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _clients[provider]


def _zero_shot_model(provider: str) -> str:
    if provider == "groq":
        return os.getenv("INTENT_GROQ_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
    return os.getenv("INTENT_OPENAI_MODEL", "gpt-4o-mini")


def classify_zero_shot(query: str, provider: str | None = None) -> str:
    """Zero-shot intent classification. Raises on any API/parse problem —
    the router catches and falls back (fail open)."""
    provider = (provider or os.getenv("INTENT_PROVIDER", "groq")).lower()
    client = _get_client(provider)
    response = client.chat.completions.create(
        model=_zero_shot_model(provider),
        messages=[
            {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=1500,
    )
    text = response.choices[0].message.content or ""
    match = re.search(r'"intent"\s*:\s*"(\w+)"', text)
    intent = (match.group(1) if match else text.strip().strip('"').lower())
    if intent not in VALID_INTENTS:
        raise ValueError(f"Zero-shot classifier returned invalid intent: {text!r}")
    return intent


# ---------------------------------------------------------------------------
# Local backend (DistilBERT + LoRA, trained offline)
# ---------------------------------------------------------------------------

_local_model = None
_local_tokenizer = None
_LOCAL_LABELS = ["clarify", "interpret", "lookup"]  # alphabetical = training order


def classify_local(query: str, model_path: str | None = None) -> str:
    """Classify with the fine-tuned DistilBERT (+ merged LoRA) model.

    Lazy-imports torch/transformers so the API deploy (which doesn't ship
    torch) never pays for them. Raises if the model directory is missing —
    the router catches and falls back.
    """
    global _local_model, _local_tokenizer
    path = model_path or os.getenv("INTENT_MODEL_PATH", "./models/intent_distilbert")
    if _local_model is None:
        from transformers import (  # noqa: PLC0415 — heavy, deliberately lazy
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        _local_tokenizer = AutoTokenizer.from_pretrained(path)
        _local_model = AutoModelForSequenceClassification.from_pretrained(path)
        _local_model.eval()

    import torch  # noqa: PLC0415

    inputs = _local_tokenizer(
        query, return_tensors="pt", truncation=True, max_length=128
    )
    with torch.no_grad():
        logits = _local_model(**inputs).logits
    idx = int(logits.argmax(dim=-1).item())
    id2label = getattr(_local_model.config, "id2label", None)
    if id2label:
        label = str(id2label[idx]).lower()
        if label in VALID_INTENTS:
            return label
    return _LOCAL_LABELS[idx]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@dataclass
class RouteDecision:
    intent: str          # lookup | interpret | clarify
    method: str          # zero_shot | local | rules | fallback:<reason>
    classifier: str      # backend that was configured


class IntentRouter:
    """Classifies queries with the configured backend, failing open to
    'interpret' (v1 behavior) on any error."""

    def __init__(self, backend: str | None = None):
        self.backend = (backend or os.getenv("INTENT_CLASSIFIER", "zero_shot")).lower()

    def classify(self, query: str) -> RouteDecision:
        if self.backend == "rules":
            return RouteDecision(
                intent=classify_rules(query), method="rules", classifier=self.backend
            )
        try:
            if self.backend == "local":
                intent = classify_local(query)
                return RouteDecision(intent=intent, method="local", classifier=self.backend)
            intent = classify_zero_shot(query)
            return RouteDecision(intent=intent, method="zero_shot", classifier=self.backend)
        except Exception:
            # Fail open: degrade to the deterministic rules classifier. If the
            # rules say "lookup" we still honor it (facts lookup itself falls
            # through to interpret on a miss), so a dead API can never make
            # the system worse than v1.
            return RouteDecision(
                intent=classify_rules(query),
                method=f"fallback_rules_from_{self.backend}",
                classifier=self.backend,
            )
