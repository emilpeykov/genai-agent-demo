from unittest.mock import MagicMock, patch

from app.react_langgraph_agent import MAX_ITERATIONS, ReActLangGraphAgent


def _mock_chat_response(message: dict):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": message}
    return mock_resp


@patch("app.react_langgraph_agent.httpx.post")
def test_answers_directly_with_no_tool_calls(mock_post):
    mock_post.return_value = _mock_chat_response(
        {"role": "assistant", "content": "The capital of France is Paris."}
    )

    agent = ReActLangGraphAgent(model="test-model")
    result = agent.handle("What's the capital of France?")

    assert result.answer == "The capital of France is Paris."
    assert result.steps == []
    assert result.iterations == 1
    assert result.hit_iteration_limit is False


@patch("app.react_langgraph_agent.httpx.post")
def test_calls_a_tool_then_answers(mock_post):
    mock_post.side_effect = [
        _mock_chat_response(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "calculator", "arguments": {"expression": "12 * 7"}}}
                ],
            }
        ),
        _mock_chat_response({"role": "assistant", "content": "12 * 7 is 84."}),
    ]

    agent = ReActLangGraphAgent(model="test-model")
    result = agent.handle("What's 12 times 7?")

    assert result.answer == "12 * 7 is 84."
    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "calculator"
    assert result.steps[0].observation == "84"
    assert result.iterations == 2


@patch("app.react_langgraph_agent.httpx.post")
def test_stops_at_max_iterations(mock_post):
    looping_response = _mock_chat_response(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "calculator", "arguments": {"expression": "1+1"}}}],
        }
    )
    mock_post.return_value = looping_response

    agent = ReActLangGraphAgent(model="test-model")
    result = agent.handle("keep going forever")

    assert result.hit_iteration_limit is True
    assert len(result.steps) == MAX_ITERATIONS


@patch("app.react_langgraph_agent.httpx.post")
def test_includes_prior_history_in_the_model_request(mock_post):
    mock_post.return_value = _mock_chat_response(
        {"role": "assistant", "content": "The root directory is '/'."}
    )

    agent = ReActLangGraphAgent(model="test-model")
    history = [
        {"role": "user", "content": "What is the current working directory?"},
        {"role": "assistant", "content": "/Users/emil/project"},
    ]
    agent.handle("and what is the root directory?", history=history)

    sent_messages = mock_post.call_args.kwargs["json"]["messages"]
    assert history[0] in sent_messages
    assert history[1] in sent_messages


@patch("app.react_langgraph_agent.httpx.post")
def test_thread_id_remembers_conversation_without_caller_supplied_history(mock_post):
    # The checkpointer-backed path: no 'history' is ever passed in by the
    # caller, yet the SECOND call (same thread_id) must still see the FIRST
    # call's user prompt and assistant answer in the messages sent to the
    # model -- the server, not the caller, is responsible for continuity.
    mock_post.side_effect = [
        _mock_chat_response({"role": "assistant", "content": "It's 84."}),
        _mock_chat_response({"role": "assistant", "content": "It's 96."}),
    ]

    agent = ReActLangGraphAgent(model="test-model")
    agent.handle("What's 12 times 7?", thread_id="user_session_123")
    agent.handle("What about 12 times 8?", thread_id="user_session_123")

    second_call_messages = mock_post.call_args_list[1].kwargs["json"]["messages"]
    contents = [m.get("content") for m in second_call_messages]
    assert "What's 12 times 7?" in contents
    assert "It's 84." in contents
    assert "What about 12 times 8?" in contents


@patch("app.react_langgraph_agent.httpx.post")
def test_thread_id_preserves_real_tool_call_args_not_just_answer_text(mock_post):
    # The key advantage over app.react_agent's caller-managed history: the
    # checkpointer persists the RAW tool_calls/arguments, so a file path (or
    # any other value learned via a tool) is recoverable later even if the
    # model's own final-answer text never literally restates it.
    mock_post.side_effect = [
        _mock_chat_response(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write_file",
                            "arguments": {
                                "file_path": "/tmp/notes.txt",
                                "content": "hello",
                            },
                        }
                    }
                ],
            }
        ),
        _mock_chat_response({"role": "assistant", "content": "I've saved it for you."}),
        _mock_chat_response({"role": "assistant", "content": "/tmp/notes.txt"}),
    ]

    agent = ReActLangGraphAgent(model="test-model")
    agent.handle("create a file with 'hello' in it", thread_id="t1")
    agent.handle("what was the exact path of that file?", thread_id="t1")

    second_call_messages = mock_post.call_args_list[2].kwargs["json"]["messages"]
    # The raw tool_calls from the FIRST turn -- not just the vague prose
    # answer "I've saved it for you" -- must still be present.
    assert any(
        m.get("tool_calls")
        and m["tool_calls"][0]["function"]["arguments"].get("file_path") == "/tmp/notes.txt"
        for m in second_call_messages
    )


@patch("app.react_langgraph_agent.httpx.post")
def test_different_thread_ids_are_isolated(mock_post):
    mock_post.side_effect = [
        _mock_chat_response({"role": "assistant", "content": "first thread answer"}),
        _mock_chat_response({"role": "assistant", "content": "second thread answer"}),
    ]

    agent = ReActLangGraphAgent(model="test-model")
    agent.handle("Remember the number 42.", thread_id="thread-a")
    agent.handle("What number did I just mention?", thread_id="thread-b")

    second_call_messages = mock_post.call_args_list[1].kwargs["json"]["messages"]
    contents = [m.get("content") for m in second_call_messages]
    assert "Remember the number 42." not in contents
    assert "What number did I just mention?" in contents


def _mock_stream_context(lines: list[str]):
    """Build a mock matching ``with httpx.stream(...) as response:`` usage,
    where ``response.iter_lines()`` yields the given NDJSON lines."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_lines.return_value = iter(lines)

    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_response
    mock_ctx.__exit__.return_value = False
    return mock_ctx


@patch("app.react_langgraph_agent.httpx.stream")
def test_handle_stream_yields_deltas_then_final_and_persists_to_checkpoint(mock_stream):
    mock_stream.return_value = _mock_stream_context(
        [
            '{"message": {"content": "The answer is "}}',
            '{"message": {"content": "84."}}',
        ]
    )

    agent = ReActLangGraphAgent(model="test-model")
    events = list(agent.handle_stream("What's 12 times 7?", thread_id="stream-thread"))

    content_deltas = [e["text"] for e in events if e["type"] == "content_delta"]
    assert content_deltas == ["The answer is ", "84."]
    final_event = events[-1]
    assert final_event["type"] == "final"
    assert final_event["result"].answer == "The answer is 84."

    # The turn must be persisted to the checkpoint for the SAME thread_id,
    # so a later handle()/handle_stream() call sees it without resending
    # history itself.
    snapshot = agent._graph.get_state({"configurable": {"thread_id": "stream-thread"}})
    contents = [m.get("content") for m in snapshot.values["messages"]]
    assert "What's 12 times 7?" in contents
    assert "The answer is 84." in contents


@patch("app.react_langgraph_agent.httpx.post")
@patch("app.react_langgraph_agent.httpx.stream")
def test_handle_stream_then_handle_share_the_same_checkpoint(mock_stream, mock_post):
    mock_stream.return_value = _mock_stream_context(['{"message": {"content": "84."}}'])
    mock_post.return_value = _mock_chat_response(
        {"role": "assistant", "content": "It was 12 times 7, which is 84."}
    )

    agent = ReActLangGraphAgent(model="test-model")
    list(agent.handle_stream("What's 12 times 7?", thread_id="mixed-thread"))
    agent.handle("What did I just ask?", thread_id="mixed-thread")

    second_call_messages = mock_post.call_args.kwargs["json"]["messages"]
    contents = [m.get("content") for m in second_call_messages]
    assert "What's 12 times 7?" in contents
    assert "84." in contents
    assert "What did I just ask?" in contents



    mock_post.return_value = _mock_chat_response({"role": "assistant", "content": "hi"})

    agent = ReActLangGraphAgent(model="qwen3:14b")
    agent.handle("hello")

    sent = mock_post.call_args.kwargs["json"]
    assert sent["model"] == "qwen3:14b"
    assert "num_ctx" in sent["options"]


@patch("app.react_langgraph_agent.httpx.post")
def test_injects_escalation_hint_after_thin_search_result(mock_post):
    # Same deterministic search-escalation behavior as
    # app.react_agent.ReActAgent, ported to the LangGraph node-based loop.
    mock_post.side_effect = [
        _mock_chat_response(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "web_search", "arguments": {"query": "OpenKBS"}}}
                ],
            }
        ),
        _mock_chat_response({"role": "assistant", "content": "I couldn't find much."}),
    ]

    def _fake_tool(name):
        tool = MagicMock()
        tool.invoke.return_value = "No web search results found for 'OpenKBS'."
        return tool

    with patch("app.react_agent._TOOLS_BY_NAME") as mock_tools:
        mock_tools.get.side_effect = _fake_tool

        agent = ReActLangGraphAgent(model="test-model")
        agent.handle("Tell me about OpenKBS")

    second_call_messages = mock_post.call_args_list[1].kwargs["json"]["messages"]
    hint_messages = [
        m["content"] for m in second_call_messages if m.get("role") == "system"
    ]
    assert any("SEARCH ESCALATION REMINDER" in c for c in hint_messages[1:])
    assert any("google_search" in c for c in hint_messages[1:])


@patch("app.react_langgraph_agent.httpx.post")
def test_asks_llm_to_verify_a_non_thin_search_result(mock_post):
    # Same "hand it back to the LLM to judge accuracy" behavior as
    # app.react_agent.ReActAgent, ported to the LangGraph node-based loop.
    mock_post.side_effect = [
        _mock_chat_response(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "wikipedia_search",
                            "arguments": {"query": "OpenKBS agentic services"},
                        }
                    }
                ],
            }
        ),
        _mock_chat_response({"role": "assistant", "content": "Here's what I found."}),
    ]

    def _fake_tool(name):
        tool = MagicMock()
        tool.invoke.return_value = (
            "Seol In-ah is a South Korean actress known for several TV dramas."
        )
        return tool

    with patch("app.react_agent._TOOLS_BY_NAME") as mock_tools:
        mock_tools.get.side_effect = _fake_tool

        agent = ReActLangGraphAgent(model="test-model")
        agent.handle("Does OpenKBS offer agentic services?")

    second_call_messages = mock_post.call_args_list[1].kwargs["json"]["messages"]
    hint_messages = [m["content"] for m in second_call_messages if m.get("role") == "system"]
    assert any("SEARCH RESULT CHECK" in c for c in hint_messages[1:])
    assert any("OpenKBS agentic services" in c for c in hint_messages[1:])


@patch("app.react_langgraph_agent.httpx.post")
def test_handle_stops_immediately_on_ask_user(mock_post):
    mock_post.side_effect = [
        _mock_chat_response(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "ask_user",
                            "arguments": {"question": "Which file do you mean?"},
                        }
                    }
                ],
            }
        ),
    ]

    agent = ReActLangGraphAgent(model="test-model")
    result = agent.handle("edit the config file", thread_id="ask-user-thread")

    assert result.answer == "Which file do you mean?"
    assert result.awaiting_user_input is True
    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "ask_user"
    assert mock_post.call_count == 1


@patch("app.react_langgraph_agent.httpx.stream")
def test_handle_stream_stops_immediately_on_ask_user(mock_stream):
    mock_stream.return_value = _mock_stream_context(
        [
            '{"message": {"content": "", "tool_calls": '
            '[{"function": {"name": "ask_user", '
            '"arguments": {"question": "Which file do you mean?"}}}]}}'
        ]
    )

    agent = ReActLangGraphAgent(model="test-model")
    events = list(agent.handle_stream("edit the config file", thread_id="ask-user-stream"))

    final_events = [e for e in events if e["type"] == "final"]
    assert len(final_events) == 1
    result = final_events[0]["result"]
    assert result.answer == "Which file do you mean?"
    assert result.awaiting_user_input is True

    # Persisted so a follow-up call on the same thread_id sees the question.
    snapshot = agent._graph.get_state({"configurable": {"thread_id": "ask-user-stream"}})
    assert snapshot.values.get("awaiting_user_input") is True

