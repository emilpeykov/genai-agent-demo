"""Application configuration and model tier definitions.

This centralizes the "model routing" cost/latency table used across the app,
simulating a real-world setup where different LLM tiers have different
price-per-1k-tokens and typical latency characteristics.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

# BUG FIX (discovered while wiring up SERPER_API_KEY): this project ships a
# '.env' file and lists 'python-dotenv' as a dependency, but nothing ever
# actually called ``load_dotenv()`` anywhere in the codebase -- so every
# value in '.env' was silently inert; only REAL shell-exported environment
# variables ever took effect. ``load_dotenv()`` populates ``os.environ`` from
# '.env' (without overriding any var already set in the real environment, so
# a real export still wins), and must run here, before any ``os.getenv(...)``
# call below. Safe to call even if no '.env' file exists (a no-op then).
from dotenv import load_dotenv

load_dotenv()

ModelTier = Literal["fast", "balanced", "powerful"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    tier: ModelTier
    cost_per_1k_tokens_usd: float
    typical_latency_ms: int
    failure_rate: float  # used by the mock provider to simulate real-world flakiness
    ollama_model: str  # local model name to use when LLM_BACKEND=ollama


# A small, realistic-looking routing table. In production this would be backed
# by a config service / feature flags so it can be tuned without redeploying.
MODEL_CATALOG: dict[ModelTier, ModelSpec] = {
    "fast": ModelSpec(
        name=os.getenv("FAST_MODEL_NAME", "gpt-4o-mini"),
        tier="fast",
        cost_per_1k_tokens_usd=0.00015,
        typical_latency_ms=350,
        failure_rate=0.03,
        ollama_model=os.getenv("OLLAMA_FAST_MODEL", "qwen2.5:7b"),
    ),
    "balanced": ModelSpec(
        name=os.getenv("BALANCED_MODEL_NAME", "gpt-4o"),
        tier="balanced",
        cost_per_1k_tokens_usd=0.0025,
        typical_latency_ms=900,
        failure_rate=0.02,
        ollama_model=os.getenv("OLLAMA_BALANCED_MODEL", "llama3.1:8b"),
    ),
    "powerful": ModelSpec(
        name=os.getenv("POWERFUL_MODEL_NAME", "o1"),
        tier="powerful",
        cost_per_1k_tokens_usd=0.015,
        typical_latency_ms=2200,
        failure_rate=0.01,
        ollama_model=os.getenv("OLLAMA_POWERFUL_MODEL", "devstral:24b"),
    ),
}

# Fallback order used when a tier fails / times out. Each tier falls back to a
# cheaper-but-still-capable neighbor rather than failing the whole request.
FALLBACK_CHAIN: dict[ModelTier, ModelTier | None] = {
    "powerful": "balanced",
    "balanced": "fast",
    "fast": None,
}

REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "8"))
MAX_RETRIES_PER_TIER = int(os.getenv("MAX_RETRIES_PER_TIER", "2"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Optional: API key for Serper.dev (https://serper.dev, free tier available),
# a thin proxy in front of real Google Search results. Only used by
# app.tools.google_search -- unset (None) means that tool reports itself
# unavailable rather than failing with a confusing auth error.
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Which provider backend to use: "mock" (default, offline/free), "ollama"
# (local models via Ollama), or "openai" (real cloud API, needs OPENAI_API_KEY).
LLM_BACKEND = os.getenv("LLM_BACKEND", "mock").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# A dedicated REASONING model, deliberately kept separate from MODEL_CATALOG's
# fast/balanced/powerful cost-tier routing table above: reasoning models
# (DeepSeek-R1, QwQ, o1-style models) aren't just "a bigger/slower model in
# the same family" -- they're trained to emit a visible chain-of-thought
# ("thinking") BEFORE their final answer, which is a genuinely different
# capability worth demonstrating on its own, not something that fits the
# cost/latency tier framing above. Defaults to deepseek-r1:14b (a good
# middle ground for a single consumer GPU/Apple Silicon machine -- the 32b
# variant is also supported by setting this env var, but needs considerably
# more RAM/VRAM headroom alongside any other already-loaded models).
OLLAMA_REASONING_MODEL = os.getenv("OLLAMA_REASONING_MODEL", "deepseek-r1:14b")


# Optional: connection string for a Postgres instance with the pgvector
# extension installed, e.g. "postgresql://user:pass@localhost:5432/mydb".
# Only used by app.postgres_rag -- unset (None) means that tool will report
# itself unavailable rather than failing the whole app at import time.
POSTGRES_DSN = os.getenv("POSTGRES_DSN")
