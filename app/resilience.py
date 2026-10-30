"""Failure-mode mitigation: retries with backoff + tier-fallback chain.

Production GenAI systems fail in predictable ways: rate limits, transient
5xxs, timeouts, or a model simply refusing/erroring on a given input. This
module implements two complementary strategies:

1. Retry the *same* tier a bounded number of times with exponential backoff
   (handles transient blips).
2. If retries are exhausted, fall back to the next cheaper tier in
   ``FALLBACK_CHAIN`` (handles sustained outages / degraded models) rather
   than failing the user's request outright.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import FALLBACK_CHAIN, MAX_RETRIES_PER_TIER, ModelTier
from app.providers import LLMProvider, LLMProviderError, LLMResponse


@dataclass
class ResilientCallResult:
    response: LLMResponse
    attempts: int
    tiers_tried: list[ModelTier] = field(default_factory=list)
    degraded: bool = False  # True if we had to fall back from the requested tier


class AllTiersFailedError(Exception):
    """Raised when the requested tier and every fallback tier failed."""


def _call_with_retries(provider: LLMProvider, prompt: str, tier: ModelTier) -> tuple[LLMResponse, int]:
    attempts = {"count": 0}

    @retry(
        reraise=True,
        stop=stop_after_attempt(MAX_RETRIES_PER_TIER),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=1),
        retry=retry_if_exception_type(LLMProviderError),
    )
    def _attempt() -> LLMResponse:
        attempts["count"] += 1
        return provider.generate(prompt, tier)

    result = _attempt()
    return result, attempts["count"]


def call_with_resilience(provider: LLMProvider, prompt: str, tier: ModelTier) -> ResilientCallResult:
    """Call ``provider`` at ``tier``, retrying and then falling back as needed."""
    tiers_tried: list[ModelTier] = []
    total_attempts = 0
    current_tier: ModelTier | None = tier
    requested_tier = tier

    last_error: Exception | None = None
    while current_tier is not None:
        tiers_tried.append(current_tier)
        try:
            response, attempts = _call_with_retries(provider, prompt, current_tier)
            total_attempts += attempts
            return ResilientCallResult(
                response=response,
                attempts=total_attempts,
                tiers_tried=tiers_tried,
                degraded=current_tier != requested_tier,
            )
        except LLMProviderError as exc:
            last_error = exc
            total_attempts += MAX_RETRIES_PER_TIER
            current_tier = FALLBACK_CHAIN.get(current_tier)

    raise AllTiersFailedError(
        f"All tiers failed for prompt (tried={tiers_tried}): {last_error}"
    ) from last_error
