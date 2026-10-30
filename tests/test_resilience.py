import pytest

from app.providers import LLMProvider, LLMProviderError, LLMResponse
from app.resilience import AllTiersFailedError, call_with_resilience


class AlwaysFailsProvider(LLMProvider):
    def generate(self, prompt: str, tier: str) -> LLMResponse:
        raise LLMProviderError(f"simulated failure at {tier}")


class FailsThenSucceedsProvider(LLMProvider):
    """Fails on 'powerful' and 'balanced' tiers, succeeds on 'fast'."""

    def generate(self, prompt: str, tier: str) -> LLMResponse:
        if tier in ("powerful", "balanced"):
            raise LLMProviderError(f"simulated failure at {tier}")
        return LLMResponse(
            text="ok",
            model="mock-fast",
            tier=tier,
            latency_ms=10,
            estimated_cost_usd=0.0001,
            tokens_used=5,
        )


class AlwaysSucceedsProvider(LLMProvider):
    def generate(self, prompt: str, tier: str) -> LLMResponse:
        return LLMResponse(
            text="ok",
            model=f"mock-{tier}",
            tier=tier,
            latency_ms=10,
            estimated_cost_usd=0.0001,
            tokens_used=5,
        )


def test_successful_call_has_no_fallback():
    result = call_with_resilience(AlwaysSucceedsProvider(), "hello", "powerful")
    assert result.degraded is False
    assert result.tiers_tried == ["powerful"]
    assert result.response.text == "ok"


def test_falls_back_through_tiers_until_success():
    result = call_with_resilience(FailsThenSucceedsProvider(), "hello", "powerful")
    assert result.degraded is True
    assert result.tiers_tried == ["powerful", "balanced", "fast"]
    assert result.response.tier == "fast"


def test_raises_when_all_tiers_fail():
    with pytest.raises(AllTiersFailedError):
        call_with_resilience(AlwaysFailsProvider(), "hello", "powerful")
