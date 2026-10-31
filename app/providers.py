"""LLM provider abstractions.

Two implementations are provided:

- ``MockLLMProvider``: a deterministic, dependency-free provider that
  simulates latency/cost/occasional failures. This lets the whole project
  (including CI) run without needing a real API key, while still exercising
  the full routing / resilience / agent logic.
- ``OpenAIChatProvider``: a thin LangChain-based wrapper used automatically
  when ``OPENAI_API_KEY`` is configured, showing how the same interface
  plugs into a real LLM backend.
"""
from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from app.config import MODEL_CATALOG, OLLAMA_BASE_URL, ModelSpec, ModelTier


@dataclass
class LLMResponse:
    text: str
    model: str
    tier: ModelTier
    latency_ms: int
    estimated_cost_usd: float
    tokens_used: int


class LLMProviderError(Exception):
    """Raised when a provider call fails (timeout, API error, rate limit, ...)."""


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, tier: ModelTier) -> LLMResponse:
        ...


def _estimate_tokens(prompt: str, completion: str) -> int:
    # Rough heuristic (~4 chars/token) good enough for cost estimation demo.
    return max(1, (len(prompt) + len(completion)) // 4)


class MockLLMProvider(LLMProvider):
    """Simulates real model behavior without any network calls.

    Useful for local dev, tests, and CI where secrets/network access aren't
    available, while still demonstrating latency and failure characteristics
    that matter for routing decisions.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def generate(self, prompt: str, tier: ModelTier) -> LLMResponse:
        spec: ModelSpec = MODEL_CATALOG[tier]

        # Simulate the occasional production failure (timeout, 5xx, rate limit).
        if self._rng.random() < spec.failure_rate:
            raise LLMProviderError(f"{spec.name} ({tier}) failed to respond")

        # Simulate latency proportional to the tier's typical latency, with jitter.
        simulated_latency_s = (spec.typical_latency_ms / 1000) * self._rng.uniform(0.8, 1.3)
        time.sleep(min(simulated_latency_s, 0.05))  # capped so tests/demo stay fast

        completion = self._fake_completion(prompt, tier)
        tokens = _estimate_tokens(prompt, completion)
        cost = round((tokens / 1000) * spec.cost_per_1k_tokens_usd, 6)

        return LLMResponse(
            text=completion,
            model=spec.name,
            tier=tier,
            latency_ms=int(simulated_latency_s * 1000),
            estimated_cost_usd=cost,
            tokens_used=tokens,
        )

    @staticmethod
    def _fake_completion(prompt: str, tier: ModelTier) -> str:
        prefix = {
            "fast": "[quick answer]",
            "balanced": "[considered answer]",
            "powerful": "[in-depth analysis]",
        }[tier]
        return f"{prefix} Response to: {prompt.strip()[:200]}"


class OpenAIChatProvider(LLMProvider):
    """Real LangChain-backed provider, used only when an API key is present."""

    def __init__(self) -> None:
        # Imported lazily so the dependency is optional for the mock-only path.
        from langchain_openai import ChatOpenAI  # type: ignore

        self._ChatOpenAI = ChatOpenAI
        self._clients: dict[str, object] = {}

    def _client(self, model_name: str):
        if model_name not in self._clients:
            self._clients[model_name] = self._ChatOpenAI(model=model_name, timeout=8)
        return self._clients[model_name]

    def generate(self, prompt: str, tier: ModelTier) -> LLMResponse:
        spec = MODEL_CATALOG[tier]
        start = time.perf_counter()
        try:
            client = self._client(spec.name)
            result = client.invoke(prompt)
            text = getattr(result, "content", str(result))
        except Exception as exc:  # pragma: no cover - network dependent
            raise LLMProviderError(str(exc)) from exc
        latency_ms = int((time.perf_counter() - start) * 1000)
        tokens = _estimate_tokens(prompt, text)
        cost = round((tokens / 1000) * spec.cost_per_1k_tokens_usd, 6)
        return LLMResponse(
            text=text,
            model=spec.name,
            tier=tier,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
            tokens_used=tokens,
        )


class OllamaProvider(LLMProvider):
    """Runs inference against a locally-hosted model via Ollama.

    Requires ``ollama serve`` running locally (default: http://localhost:11434)
    with the relevant models pulled (e.g. ``ollama pull llama3.1:8b``).
    Since the model runs on your own machine, there's no per-token API cost --
    ``estimated_cost_usd`` is always 0.0, which is itself a useful data point
    when discussing cost/latency tradeoffs (free but uses local compute/time).
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL) -> None:
        self._base_url = base_url.rstrip("/")

    def generate(self, prompt: str, tier: ModelTier) -> LLMResponse:
        spec = MODEL_CATALOG[tier]
        start = time.perf_counter()
        try:
            response = httpx.post(
                f"{self._base_url}/api/generate",
                json={"model": spec.ollama_model, "prompt": prompt, "stream": False},
                timeout=120,  # local generation can be slow on CPU-only machines
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise LLMProviderError(
                f"Could not reach Ollama at {self._base_url} for model "
                f"'{spec.ollama_model}': {exc}"
            ) from exc

        text = data.get("response", "")
        latency_ms = int((time.perf_counter() - start) * 1000)
        # Ollama reports actual token counts when available; fall back to the estimate.
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        if not tokens:
            tokens = _estimate_tokens(prompt, text)

        return LLMResponse(
            text=text,
            model=spec.ollama_model,
            tier=tier,
            latency_ms=latency_ms,
            estimated_cost_usd=0.0,
            tokens_used=tokens,
        )


def get_default_provider() -> LLMProvider:
    from app.config import LLM_BACKEND, OPENAI_API_KEY

    if LLM_BACKEND == "ollama":
        return OllamaProvider()

    if LLM_BACKEND == "openai" and OPENAI_API_KEY:
        try:
            return OpenAIChatProvider()
        except ImportError:  # langchain-openai not installed
            pass

    return MockLLMProvider()
