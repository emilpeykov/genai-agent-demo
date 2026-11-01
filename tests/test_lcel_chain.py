from unittest.mock import MagicMock
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.prompts import PromptTemplate

from app.lcel_chain import (
    build_assistant_chain,
    build_prompt_model_parser_chain,
    build_tool_calling_chain,
    build_tool_calling_chain_streaming,
    build_tool_calling_chain_streaming_with_memory,
    build_tool_calling_chain_with_memory,
    get_session_history,
)
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


def test_get_session_history_returns_same_instance_for_same_session_id():
    session_id = str(uuid4())

    assert get_session_history(session_id) is get_session_history(session_id)


def test_get_session_history_returns_different_instances_for_different_ids():
    assert get_session_history(str(uuid4())) is not get_session_history(str(uuid4()))


def test_build_tool_calling_chain_with_memory_remembers_prior_turns():
    llm_with_tools = MagicMock()
    llm_with_tools.invoke.return_value = AIMessage(content="ok", tool_calls=[])

    chain = build_tool_calling_chain_with_memory(llm_with_tools=llm_with_tools)
    config = {"configurable": {"session_id": str(uuid4())}}

    chain.invoke({"user_request": "my name is Bob"}, config=config)
    chain.invoke({"user_request": "what is my name?"}, config=config)

    # The SECOND call's messages must include the FIRST turn's human message
    # and the model's own first reply -- i.e. real memory, not a stateless
    # single-turn call.
    second_call_messages = llm_with_tools.invoke.call_args_list[1][0][0]
    human_texts = [
        m.content for m in second_call_messages if m.__class__.__name__ == "HumanMessage"
    ]
    assert human_texts == ["my name is Bob", "what is my name?"]


def test_build_tool_calling_chain_with_memory_isolates_different_sessions():
    llm_with_tools = MagicMock()
    llm_with_tools.invoke.return_value = AIMessage(content="ok", tool_calls=[])

    chain = build_tool_calling_chain_with_memory(llm_with_tools=llm_with_tools)
    session_a = {"configurable": {"session_id": str(uuid4())}}
    session_b = {"configurable": {"session_id": str(uuid4())}}

    chain.invoke({"user_request": "secret for session A"}, config=session_a)
    chain.invoke({"user_request": "what did I say before?"}, config=session_b)

    # Session B's call must NOT see anything said in session A.
    session_b_call_messages = llm_with_tools.invoke.call_args_list[1][0][0]
    human_texts = [
        m.content for m in session_b_call_messages if m.__class__.__name__ == "HumanMessage"
    ]
    assert human_texts == ["what did I say before?"]


def test_build_tool_calling_chain_streaming_yields_chunks_when_no_tool_call():
    llm_with_tools = MagicMock()
    llm_with_tools.stream.return_value = [
        AIMessageChunk(content="Hello"),
        AIMessageChunk(content=", world"),
        AIMessageChunk(content="!"),
    ]

    chain = build_tool_calling_chain_streaming(llm_with_tools=llm_with_tools)
    chunks = list(chain.stream({"user_request": "hi"}))

    assert chunks == ["Hello", ", world", "!"]
    llm_with_tools.stream.assert_called_once()


def test_build_tool_calling_chain_streaming_invoke_concatenates_full_answer():
    llm_with_tools = MagicMock()
    llm_with_tools.stream.return_value = [
        AIMessageChunk(content="partial "),
        AIMessageChunk(content="answer"),
    ]

    chain = build_tool_calling_chain_streaming(llm_with_tools=llm_with_tools)
    result = chain.invoke({"user_request": "hi"})

    assert result == "partial answer"


def test_build_tool_calling_chain_streaming_executes_tool_then_streams_final_answer():
    llm_with_tools = MagicMock()
    first_round = [
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": "calculator", "args": '{"expression": "2 + 2"}', "id": "call1", "index": 0}
            ],
        )
    ]
    second_round = [AIMessageChunk(content="4")]
    llm_with_tools.stream.side_effect = [first_round, second_round]

    chain = build_tool_calling_chain_streaming(llm_with_tools=llm_with_tools)
    chunks = list(chain.stream({"user_request": "what is 2 + 2?"}))

    assert chunks == ["4"]
    assert llm_with_tools.stream.call_count == 2
    second_call_messages = llm_with_tools.stream.call_args_list[1][0][0]
    tool_messages = [m for m in second_call_messages if m.__class__.__name__ == "ToolMessage"]
    assert tool_messages[0].content == "4"


def test_build_tool_calling_chain_streaming_with_memory_remembers_prior_turns():
    llm_with_tools = MagicMock()
    llm_with_tools.stream.return_value = [AIMessageChunk(content="ok")]

    chain = build_tool_calling_chain_streaming_with_memory(llm_with_tools=llm_with_tools)
    config = {"configurable": {"session_id": str(uuid4())}}

    list(chain.stream({"user_request": "my name is Bob"}, config=config))
    list(chain.stream({"user_request": "what is my name?"}, config=config))

    second_call_messages = llm_with_tools.stream.call_args_list[1][0][0]
    human_texts = [
        m.content for m in second_call_messages if m.__class__.__name__ == "HumanMessage"
    ]
    assert human_texts == ["my name is Bob", "what is my name?"]

