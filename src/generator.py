"""
src/generator.py — LLM generation layer for RAG Engineering Assistant.

Responsibilities:
    - Build citation-grounded prompt from retrieved chunks
    - Call the configured LLM: Groq (free tier, default), OpenAI GPT-4o-mini,
      or Ollama Llama 3 — selected via the LLM_PROVIDER env var
    - Stream responses via async generator (SSE-compatible)
    - Track token usage and estimated cost
    - Detect out-of-corpus queries and handle gracefully

Usage:
    from src.generator import Generator
    gen = Generator()

    # Non-streaming
    response = gen.generate(query="...", chunks=[...])
    print(response["answer"])

    # Streaming (async, for FastAPI SSE)
    async for token in gen.stream(query="...", chunks=[...]):
        print(token, end="", flush=True)
"""

from __future__ import annotations

import os
from typing import AsyncIterator

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

REFUSAL_PHRASE = "I don't have information on that in the provided documents."

SYSTEM_PROMPT = f"""You are a technical assistant for engineers. Your job is to answer
questions accurately and concisely based ONLY on the document excerpts provided.

Rules:
1. Answer ONLY from the provided document excerpts. Do not use outside knowledge.
2. If the answer is not in the excerpts, say clearly: "{REFUSAL_PHRASE}" Do not guess.
3. Always cite your source using this format: [Source: {{document_name}}, Page {{page}}]
4. If multiple excerpts support the answer, cite all of them.
5. Keep answers concise but complete. Use numbered lists for multi-step answers.
6. For numerical specifications (dimensions, pressures, temperatures), be exact.
   Do not round or paraphrase numbers."""


def is_refusal(answer: str) -> bool:
    """Return True when the answer contains the instructed refusal phrase."""
    return REFUSAL_PHRASE.lower() in (answer or "").lower()


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build the user prompt by combining retrieved chunks with the query.

    Each chunk is prefaced with its source citation so the LLM can reference it.
    """
    if not chunks:
        context = "(No document excerpts retrieved — answer cannot be grounded.)"
    else:
        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "Unknown")
            page = chunk.get("page", "?")
            context_parts.append(
                f"[Excerpt {i} — Source: {source}, Page {page}]\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

    return f"Document excerpts:\n\n{context}\n\n---\n\nQuestion: {query}\n\nAnswer:"


# ---------------------------------------------------------------------------
# LLM client — OpenAI and Groq share the same OpenAI-compatible API
# ---------------------------------------------------------------------------
#
# Groq is a drop-in OpenAI-compatible endpoint: identical SDK and request shape,
# just a different base_url and API key. That lets us swap the (paid) OpenAI
# generator for Groq's free tier without changing any request/response code.

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_llm_clients: dict = {}


def _get_llm_client(provider: str):
    """
    Return a cached OpenAI-compatible client for the given provider.

        "openai" → OpenAI API   (OPENAI_API_KEY)
        "groq"   → Groq API      (GROQ_API_KEY) — OpenAI-compatible, free tier
    """
    if provider not in _llm_clients:
        from openai import OpenAI
        if provider == "groq":
            _llm_clients[provider] = OpenAI(
                base_url=_GROQ_BASE_URL,
                api_key=os.getenv("GROQ_API_KEY"),
            )
        else:  # openai
            _llm_clients[provider] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _llm_clients[provider]


# Pricing as of 2026 (gpt-4o-mini). Groq's free tier and local Ollama cost $0,
# so a dollar estimate is only produced for the OpenAI provider.
_COST_PER_1M_INPUT = 0.15    # USD
_COST_PER_1M_OUTPUT = 0.60   # USD


def _estimate_cost(provider: str, prompt_tokens: int, completion_tokens: int) -> float:
    if provider != "openai":
        return 0.0
    return (prompt_tokens * _COST_PER_1M_INPUT + completion_tokens * _COST_PER_1M_OUTPUT) / 1_000_000


# ---------------------------------------------------------------------------
# Ollama client (local LLM)
# ---------------------------------------------------------------------------

def _generate_ollama(messages: list[dict]) -> dict:
    """Generate a response using a local Ollama model."""
    import httpx
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    response = httpx.post(
        f"{base_url}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "answer": data["message"]["content"],
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
        "cost_usd": 0.0,  # local LLM = free
        "model": model,
        "provider": "ollama",
    }


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class Generator:
    """
    LLM generation wrapper.

    Provider is controlled by LLM_PROVIDER env var:
        "openai" (default) → GPT-4o-mini via OpenAI API
        "ollama"           → Local model via Ollama
    """

    # Default model per provider. Override with the `model` arg or the
    # matching env var (OPENAI_MODEL / GROQ_MODEL / OLLAMA_MODEL).
    # Groq model IDs change — check https://console.groq.com/docs/models.
    _DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "groq": "openai/gpt-oss-20b",
        "ollama": "llama3.1:8b",
    }

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        env_model = {
            "openai": os.getenv("OPENAI_MODEL"),
            "groq": os.getenv("GROQ_MODEL"),
            "ollama": os.getenv("OLLAMA_MODEL"),
        }.get(self.provider)
        self.model = model or env_model or self._DEFAULT_MODELS.get(self.provider, "gpt-4o-mini")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, query: str, chunks: list[dict]) -> dict:
        """
        Generate a grounded answer for the query.

        Returns:
            {
                "answer": str,
                "sources": list[{"source": str, "page": int}],
                "chunks_used": int,
                "prompt_tokens": int,
                "completion_tokens": int,
                "cost_usd": float,
                "model": str,
                "provider": str,
            }
        """
        prompt = build_prompt(query, chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        if self.provider == "ollama":
            result = _generate_ollama(messages)
        else:
            # openai + groq share the OpenAI-compatible chat API
            client = _get_llm_client(self.provider)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            answer = response.choices[0].message.content
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            result = {
                "answer": answer,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": _estimate_cost(self.provider, prompt_tokens, completion_tokens),
                "model": self.model,
                "provider": self.provider,
            }

        # Extract cited sources from chunks
        sources = [
            {"source": c.get("source", ""), "page": c.get("page", 0)}
            for c in chunks
        ]
        result["sources"] = sources
        result["chunks_used"] = len(chunks)
        result["refused"] = is_refusal(result.get("answer", ""))
        return result

    async def stream(
        self,
        query: str,
        chunks: list[dict],
    ) -> AsyncIterator[str]:
        """
        Stream the response token by token (async generator).

        Yields individual text tokens as strings.
        The final token is a JSON blob prefixed with "__METADATA__:" containing
        sources, cost, and token counts so the frontend can display them.

        Usage in FastAPI:
            from fastapi.responses import StreamingResponse
            async def gen():
                async for token in generator.stream(query, chunks):
                    yield f"data: {token}\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        """
        import json

        prompt = build_prompt(query, chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        prompt_tokens = 0
        completion_tokens = 0
        answer_parts: list[str] = []

        if self.provider == "ollama":
            # Ollama streaming not implemented — fall back to non-streaming
            result = _generate_ollama(messages)
            prompt_tokens = result.get("prompt_tokens", 0)
            completion_tokens = result.get("completion_tokens", 0)
            answer_parts.append(result["answer"])
            for word in result["answer"].split(" "):
                yield word + " "
        else:
            # openai + groq share the OpenAI-compatible streaming API
            client = _get_llm_client(self.provider)
            stream = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for chunk in stream:
                # The final usage chunk can arrive with an empty choices list
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        answer_parts.append(delta.content)
                        yield delta.content
                        completion_tokens += 1
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens

        # Emit metadata at end of stream for the frontend to parse.
        # IMPORTANT: no leading "\n" here. The API wraps each yield as an SSE
        # "data: <token>\n\n" frame, so a leading newline would split the line
        # and the browser's EventSource would drop the metadata as an unknown
        # field — leaving the Run Stats / citations / chunk panels empty.
        sources = [
            {"source": c.get("source", ""), "page": c.get("page", 0)}
            for c in chunks
        ]
        chunk_previews = [
            {
                "text": (c["text"][:300] + "...") if len(c.get("text", "")) > 300 else c.get("text", ""),
                "source": c.get("source", ""),
                "page": c.get("page", 0),
                "retrieval_method": c.get("retrieval_method"),
                "rrf_score": c.get("rrf_score"),
                "rerank_score": c.get("rerank_score"),
            }
            for c in chunks
        ]
        metadata = {
            "sources": sources,
            "chunks_used": len(chunks),
            "chunks": chunk_previews,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": _estimate_cost(self.provider, prompt_tokens, completion_tokens),
            "model": self.model,
            "provider": self.provider,
            "refused": is_refusal("".join(answer_parts)),
        }
        yield f"__METADATA__:{json.dumps(metadata)}"
