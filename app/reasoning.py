"""A dedicated REASONING model path (DeepSeek-R1 via Ollama), separate from
the fast/balanced/powerful cost-tier routing in ``app.providers``/``app.config``.

Why this is its own module instead of just another ``ModelTier``: reasoning
models like DeepSeek-R1 are trained to emit a visible chain-of-thought
("thinking") BEFORE their final answer -- a genuinely different, inspectable
capability worth surfacing on its own, not a cost/latency variant of the same
request/response shape every other tier uses. Showing the raw reasoning
trace alongside the final answer is itself a useful AI-engineering
demonstration (e.g. for debugging why a model reached a conclusion, or
building user-facing "show your work" UIs).

Mechanics: Ollama's ``/api/chat`` endpoint supports a ``"think": true``
request flag for reasoning-capable models (DeepSeek-R1, QwQ, etc.), which
returns the chain-of-thought in a separate ``message.thinking`` field,
cleanly split from ``message.content`` (the final answer). Older Ollama
versions / some models instead inline the reasoning as a literal
``<think>...</think>`` block inside ``message.content`` -- ``_split_think_tags``
is a fallback for that case, so this module behaves consistently either way.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

from app.config import OLLAMA_BASE_URL, OLLAMA_REASONING_MODEL

_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class ReasoningResult:
    reasoning: str
    answer: str
    model: str
    latency_ms: int


def _split_think_tags(content: str) -> tuple[str, str]:
    """Fallback parser for models/Ollama versions that inline the
    chain-of-thought as a literal ``<think>...</think>`` block inside the
    message content instead of a separate ``thinking`` field. Returns
    ``(reasoning, answer)`` with the tag stripped out of the answer either
    way. If no ``<think>`` block is present, ``reasoning`` is empty and
    ``answer`` is the content unchanged.
    """
    match = _THINK_BLOCK_RE.search(content)
    if not match:
        return "", content.strip()
    reasoning = match.group(1).strip()
    answer = (content[: match.start()] + content[match.end() :]).strip()
    return reasoning, answer


def generate_with_reasoning(
    prompt: str,
    model: str | None = None,
    base_url: str = OLLAMA_BASE_URL,
    timeout: float = 180,
) -> ReasoningResult:
    """Call a reasoning-capable local model and return its chain-of-thought
    and final answer SEPARATELY.

    Uses a generous default timeout (180s): reasoning models spend real
    wall-clock time "thinking" before producing a final answer, often far
    longer than a normal single-pass completion -- this is expected,
    successful behavior for this model family, not a sign of a hung request.
    """
    resolved_model = model or OLLAMA_REASONING_MODEL
    start = time.perf_counter()
    response = httpx.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": resolved_model,
            "messages": [{"role": "user", "content": prompt}],
            "think": True,
            "stream": False,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    latency_ms = int((time.perf_counter() - start) * 1000)

    message = data.get("message", {}) or {}
    reasoning = message.get("thinking") or ""
    content = message.get("content") or ""

    if not reasoning:
        # Either an Ollama version/model that doesn't support the "think"
        # flag, or it inlined the chain-of-thought as a <think> block anyway.
        reasoning, content = _split_think_tags(content)

    return ReasoningResult(
        reasoning=reasoning, answer=content.strip(), model=resolved_model, latency_ms=latency_ms
    )


def stream_with_reasoning(
    prompt: str,
    model: str | None = None,
    base_url: str = OLLAMA_BASE_URL,
    timeout: float = 180,
) -> Iterator[tuple[str, str]]:
    """Streaming counterpart to ``generate_with_reasoning``.

    Yields ``(kind, text)`` tuples as chunks arrive, where ``kind`` is either
    ``"reasoning"`` (chain-of-thought tokens) or ``"answer"`` (final-answer
    tokens) -- callers (e.g. the FastAPI streaming endpoint, the Gradio tab)
    decide how to render each kind (e.g. a collapsed "thinking..." panel vs.
    the main answer text) without needing to re-parse ``<think>`` tags
    themselves on every chunk.
    """
    resolved_model = model or OLLAMA_REASONING_MODEL
    with httpx.stream(
        "POST",
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": resolved_model,
            "messages": [{"role": "user", "content": prompt}],
            "think": True,
            "stream": True,
        },
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            message = chunk.get("message", {}) or {}
            if message.get("thinking"):
                yield "reasoning", message["thinking"]
            if message.get("content"):
                yield "answer", message["content"]

