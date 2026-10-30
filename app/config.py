"""Application configuration and model tier definitions.

This centralizes the "model routing" cost/latency table used across the app,
simulating a real-world setup where different LLM tiers have different
price-per-1k-tokens and typical latency characteristics.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

ModelTier = Literal["fast", "balanced", "powerful"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    tier: ModelTier
    cost_per_1k_tokens_usd: float
    typical_latency_ms: int
    failure_rate: float  # used by the mock provider to simulate real-world flakiness


# A small, realistic-looking routing table. In production this would be backed
# by a config service / feature flags so it can be tuned without redeploying.
MODEL_CATALOG: dict[ModelTier, ModelSpec] = {
    "fast": ModelSpec(
        name=os.getenv("FAST_MODEL_NAME", "gpt-4o-mini"),
        tier="fast",
        cost_per_1k_tokens_usd=0.00015,
        typical_latency_ms=350,
        failure_rate=0.03,
    ),
    "balanced": ModelSpec(
        name=os.getenv("BALANCED_MODEL_NAME", "gpt-4o"),
        tier="balanced",
        cost_per_1k_tokens_usd=0.0025,
        typical_latency_ms=900,
        failure_rate=0.02,
    ),
    "powerful": ModelSpec(
        name=os.getenv("POWERFUL_MODEL_NAME", "o1"),
        tier="powerful",
        cost_per_1k_tokens_usd=0.015,
        typical_latency_ms=2200,
        failure_rate=0.01,
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
