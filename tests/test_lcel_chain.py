from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate

from app.lcel_chain import build_assistant_chain, build_prompt_model_parser_chain, build_tool_calling_chain
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


def test_build_prompt_model_parser_chain_renders_arbitrary_template():
    provider = _fake_provider("generic chain answer")
    template = PromptTemplate.from_template("Translate to French: {text}")

    chain = build_prompt_model_parser_chain(template, provider=provider, tier="balanced")
    result = chain.invoke({"text": "hello"})

    assert result == "generic chain answer"
    sent_prompt, sent_tier = provider.generate.call_args[0]
    assert "Translate to French: hello" == sent_prompt
    assert sent_tier == "balanced"


def test_build_prompt_model_parser_chain_supports_custom_output_parser():
    from langchain_core.output_parsers import BaseOutputParser

    class _UpperParser(BaseOutputParser[str]):
        def parse(self, text: str) -> str:
            return text.upper()

    provider = _fake_provider("lower case text")
    template = PromptTemplate.from_template("{text}")

    chain = build_prompt_model_parser_chain(
        template, provider=provider, output_parser=_UpperParser()
    )
    result = chain.invoke({"text": "hello"})

    assert result == "LOWER CASE TEXT"


def test_build_tool_calling_chain_answers_directly_when_no_tool_call():
    llm_with_tools = MagicMock()
    llm_with_tools.invoke.return_value = AIMessage(content="plain answer", tool_calls=[])

    chain = build_tool_calling_chain(llm_with_tools=llm_with_tools)
    result = chain.invoke({"user_request": "hello"})

    assert result == "plain answer"
    llm_with_tools.invoke.assert_called_once()


def test_build_tool_calling_chain_executes_tool_and_returns_final_answer():
    llm_with_tools = MagicMock()
    first_response = AIMessage(
        content="",
        tool_calls=[{"name": "calculator", "args": {"expression": "2 + 2"}, "id": "call1"}],
    )
    second_response = AIMessage(content="4", tool_calls=[])
    llm_with_tools.invoke.side_effect = [first_response, second_response]

    chain = build_tool_calling_chain(llm_with_tools=llm_with_tools)
    result = chain.invoke({"user_request": "what is 2 + 2?"})

    assert result == "4"
    assert llm_with_tools.invoke.call_count == 2
    # The real calculator tool ran -- its result was fed back as a ToolMessage
    # in the second call's message history.
    second_call_messages = llm_with_tools.invoke.call_args_list[1][0][0]
    tool_messages = [m for m in second_call_messages if m.__class__.__name__ == "ToolMessage"]
    assert tool_messages[0].content == "4"


def test_build_tool_calling_chain_handles_unknown_tool_name():
    llm_with_tools = MagicMock()
    first_response = AIMessage(
        content="",
        tool_calls=[{"name": "not_a_real_tool", "args": {}, "id": "call1"}],
    )
    second_response = AIMessage(content="final answer", tool_calls=[])
    llm_with_tools.invoke.side_effect = [first_response, second_response]

    chain = build_tool_calling_chain(llm_with_tools=llm_with_tools)
    result = chain.invoke({"user_request": "hello"})

    assert result == "final answer"
    second_call_messages = llm_with_tools.invoke.call_args_list[1][0][0]
    tool_messages = [m for m in second_call_messages if m.__class__.__name__ == "ToolMessage"]
    assert "unknown tool" in tool_messages[0].content


def test_build_tool_calling_chain_stops_after_max_rounds():
    llm_with_tools = MagicMock()
    # Always asks for a tool -- the loop must bail out via the round cap
    # instead of looping forever.
    llm_with_tools.invoke.return_value = AIMessage(
        content="still thinking",
        tool_calls=[{"name": "calculator", "args": {"expression": "1 + 1"}, "id": "call1"}],
    )

    chain = build_tool_calling_chain(llm_with_tools=llm_with_tools)
    result = chain.invoke({"user_request": "hello"})

    assert result == "still thinking"
    assert llm_with_tools.invoke.call_count == 3  # MAX_TOOL_ROUNDS
