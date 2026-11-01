from unittest.mock import MagicMock

from app.lcel_chain import build_assistant_chain
from app.providers import LLMResponse


def _fake_provider(response_text: str = "a mocked answer") -> MagicMock:
    provider = MagicMock()
    provider.generate.return_value = LLMResponse(
        text=response_text,
        model="test-model",
        tier="fast",
        latency_ms=10,
        estimated_cost_usd=0.0,
        tokens_used=5,
    )
    return provider


def test_build_assistant_chain_invokes_provider_with_rendered_prompt():
    provider = _fake_provider()

    chain = build_assistant_chain(provider=provider, tier="fast")
    result = chain.invoke({"user_request": "What is the refund policy?"})

    assert result == "a mocked answer"
    provider.generate.assert_called_once()
    sent_prompt, sent_tier = provider.generate.call_args[0]
    assert "What is the refund policy?" in sent_prompt
    assert "User request:" in sent_prompt
    assert sent_tier == "fast"


def test_build_assistant_chain_bakes_in_company_name_and_date():
    from datetime import date

    provider = _fake_provider()
    chain = build_assistant_chain(provider=provider, tier="fast")

    chain.invoke({"user_request": "hello"})

    sent_prompt = provider.generate.call_args[0][0]
    assert "Acme GenAI Demo Co." in sent_prompt
    assert date.today().isoformat() in sent_prompt


def test_build_assistant_chain_uses_requested_tier():
    provider = _fake_provider()
    chain = build_assistant_chain(provider=provider, tier="powerful")

    chain.invoke({"user_request": "hello"})

    sent_tier = provider.generate.call_args[0][1]
    assert sent_tier == "powerful"


def test_build_assistant_chain_returns_plain_string_output():
    provider = _fake_provider("plain text response")
    chain = build_assistant_chain(provider=provider)

    result = chain.invoke({"user_request": "anything"})

    assert isinstance(result, str)
    assert result == "plain text response"
