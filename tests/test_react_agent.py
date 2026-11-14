from unittest.mock import MagicMock, patch

from app.react_agent import MAX_ITERATIONS, ReActAgent


def _mock_chat_response(message: dict):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": message}
    return mock_resp


@patch("app.react_agent.httpx.post")
def test_react_agent_answers_directly_with_no_tool_calls(mock_post):
    mock_post.return_value = _mock_chat_response(
        {"role": "assistant", "content": "The capital of France is Paris."}
    )

    agent = ReActAgent(model="test-model")
    result = agent.handle("What's the capital of France?")

    assert result.answer == "The capital of France is Paris."
    assert result.steps == []
    assert result.iterations == 1
    assert result.hit_iteration_limit is False


@patch("app.react_agent.httpx.post")
def test_react_agent_calls_a_tool_then_answers(mock_post):
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

    agent = ReActAgent(model="test-model")
    result = agent.handle("What's 12 times 7?")

    assert result.answer == "12 * 7 is 84."
    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "calculator"
    assert result.steps[0].observation == "84"
    assert result.iterations == 2


@patch("app.react_agent.httpx.post")
def test_react_agent_handles_json_string_arguments(mock_post):
    # Some models emit tool call arguments as a JSON string rather than a dict.
    mock_post.side_effect = [
        _mock_chat_response(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "calculator", "arguments": '{"expression": "2 + 2"}'}}
                ],
            }
        ),
        _mock_chat_response({"role": "assistant", "content": "It's 4."}),
    ]

    agent = ReActAgent(model="test-model")
    result = agent.handle("what is 2+2")

    assert result.steps[0].observation == "4"
    assert result.answer == "It's 4."


@patch("app.react_agent.httpx.post")
def test_react_agent_stops_at_max_iterations(mock_post):
    looping_response = _mock_chat_response(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "calculator", "arguments": {"expression": "1+1"}}}],
        }
    )
    mock_post.return_value = looping_response

    agent = ReActAgent(model="test-model")
    result = agent.handle("keep going forever")

    assert result.hit_iteration_limit is True
    assert len(result.steps) == MAX_ITERATIONS


@patch("app.react_agent.httpx.post")
def test_react_agent_handles_unknown_tool_name_gracefully(mock_post):
    mock_post.side_effect = [
        _mock_chat_response(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "nonexistent_tool", "arguments": {}}}],
            }
        ),
        _mock_chat_response({"role": "assistant", "content": "Sorry, I couldn't do that."}),
    ]

    agent = ReActAgent(model="test-model")
    result = agent.handle("do something weird")

    assert "unknown tool" in result.steps[0].observation


@patch("app.react_agent.httpx.post")
def test_react_agent_includes_prior_history_in_the_model_request(mock_post):
    # Regression test: without passing `history`, every call was completely
    # stateless -- a follow-up like "what about that?" made no sense to the
    # model because it never saw the earlier turns of the conversation.
    mock_post.return_value = _mock_chat_response(
        {"role": "assistant", "content": "The root directory is '/'."}
    )

    agent = ReActAgent(model="test-model")
    history = [
        {"role": "user", "content": "What is the current working directory?"},
        {"role": "assistant", "content": "/Users/emil/project"},
    ]
    agent.handle("and what is the root directory?", history=history)

    sent_messages = mock_post.call_args.kwargs["json"]["messages"]
    # system prompt, then the two history turns, then the new user prompt.
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[1] == history[0]
    assert sent_messages[2] == history[1]
    assert sent_messages[3] == {"role": "user", "content": "and what is the root directory?"}


@patch("app.react_agent.httpx.post")
def test_react_agent_works_without_history(mock_post):
    mock_post.return_value = _mock_chat_response({"role": "assistant", "content": "Paris."})

    agent = ReActAgent(model="test-model")
    agent.handle("What's the capital of France?")

    sent_messages = mock_post.call_args.kwargs["json"]["messages"]
    assert len(sent_messages) == 2  # just system + the new user turn


@patch("app.react_agent.httpx.post")
def test_react_agent_feeds_validation_error_back_so_model_can_self_correct(mock_post):    # Pro-tip pattern: client-side (pydantic) validation catches a malformed
    # tool call -- here, read_file missing its required 'file_path' -- BEFORE
    # the real tool ever runs, and the precise validation error is fed back
    # as the observation. The model then reads its own mistake and corrects
    # the call on its second pass, exactly like a normal tool round-trip.
    mock_post.side_effect = [
        _mock_chat_response(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": {}}}],
            }
        ),
        _mock_chat_response(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"file_path": "/tmp/x.txt"},
                        }
                    }
                ],
            }
        ),
        _mock_chat_response({"role": "assistant", "content": "The file says hello."}),
    ]

    with patch("app.react_agent._TOOLS_BY_NAME") as mock_tools:
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "hello"
        mock_tools.get.side_effect = lambda name: mock_tool

        agent = ReActAgent(model="test-model")
        result = agent.handle("read /tmp/x.txt")

    assert len(result.steps) == 2
    assert "invalid arguments for tool 'read_file'" in result.steps[0].observation
    assert "file_path" in result.steps[0].observation
    assert result.steps[1].observation == "hello"
    assert result.answer == "The file says hello."
    # The first (invalid) call must never have reached the real tool.
    mock_tool.invoke.assert_called_once_with({"file_path": "/tmp/x.txt"})


@patch("app.react_agent.httpx.post")
def test_react_agent_injects_escalation_hint_after_thin_search_result(mock_post):
    # Regression test: asked to research a topic, 'web_search' returned a
    # thin/empty result and the model gave up without ever trying
    # 'google_search'/'fetch_webpage'. The loop must deterministically inject
    # a system reminder naming the untried search tools right after a thin
    # search observation -- not rely purely on the system prompt's wording.
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
        _mock_chat_response(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "google_search", "arguments": {"query": "OpenKBS"}}}
                ],
            }
        ),
        _mock_chat_response({"role": "assistant", "content": "Here's what I found."}),
    ]

    def _fake_tool(name):
        tool = MagicMock()
        if name == "web_search":
            tool.invoke.return_value = "No web search results found for 'OpenKBS'."
        elif name == "google_search":
            tool.invoke.return_value = "OpenKBS is a real thing with plenty of detail."
        return tool

    with patch("app.react_agent._TOOLS_BY_NAME") as mock_tools:
        mock_tools.get.side_effect = _fake_tool

        agent = ReActAgent(model="test-model")
        result = agent.handle("Tell me about OpenKBS")

    assert len(result.steps) == 2
    assert result.steps[0].tool_name == "web_search"
    assert result.steps[1].tool_name == "google_search"

    # The SECOND model call (made right after web_search's thin observation)
    # must have seen a deterministic escalation reminder injected right
    # after that observation, naming 'google_search' as untried.
    third_call_messages = mock_post.call_args_list[1].kwargs["json"]["messages"]
    hint_messages = [
        m["content"] for m in third_call_messages if m.get("role") == "system"
    ]
    assert any("SEARCH ESCALATION REMINDER" in c for c in hint_messages[1:])
    assert any("google_search" in c for c in hint_messages[1:])


@patch("app.react_agent.httpx.post")
def test_react_agent_does_not_inject_hint_after_a_good_search_result(mock_post):
    # Counter-test: a search tool returning a REAL, substantive result must
    # NOT trigger the escalation reminder.
    mock_post.side_effect = [
        _mock_chat_response(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "web_search", "arguments": {"query": "Python"}}}
                ],
            }
        ),
        _mock_chat_response({"role": "assistant", "content": "Python is a language."}),
    ]

    with patch("app.react_agent._TOOLS_BY_NAME") as mock_tools:
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "Python is a popular programming language used for..."
        mock_tools.get.side_effect = lambda name: mock_tool

        agent = ReActAgent(model="test-model")
        agent.handle("What is Python?")

    second_call_messages = mock_post.call_args_list[1].kwargs["json"]["messages"]
    assert not any(
        "SEARCH ESCALATION REMINDER" in m.get("content", "") for m in second_call_messages
    )


@patch("app.react_agent.httpx.post")
def test_react_agent_asks_llm_to_verify_a_non_thin_search_result(mock_post):
    # A non-empty, non-error search result is NOT automatically trusted --
    # the agent must hand it back to the LLM itself with an explicit
    # "SEARCH RESULT CHECK" asking it to judge accuracy/relevance before
    # using it, rather than us guessing relevance ourselves.
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

    with patch("app.react_agent._TOOLS_BY_NAME") as mock_tools:
        mock_tool = MagicMock()
        # A real, substantive, but UNRELATED result (not empty/error, so
        # is_thin_search_result alone wouldn't catch it).
        mock_tool.invoke.return_value = (
            "Seol In-ah is a South Korean actress known for several TV dramas."
        )
        mock_tools.get.side_effect = lambda name: mock_tool

        agent = ReActAgent(model="test-model")
        agent.handle("Does OpenKBS offer agentic services?")

    second_call_messages = mock_post.call_args_list[1].kwargs["json"]["messages"]
    hint_messages = [m["content"] for m in second_call_messages if m.get("role") == "system"]
    assert any("SEARCH RESULT CHECK" in c for c in hint_messages[1:])
    assert any("OpenKBS agentic services" in c for c in hint_messages[1:])


@patch("app.react_agent.httpx.post")
def test_react_agent_stops_immediately_on_ask_user(mock_post):
    # 'ask_user' must deterministically stop the WHOLE loop right there --
    # the question becomes this turn's "answer" and awaiting_user_input is
    # set, rather than looping back to the model for another round.
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

    agent = ReActAgent(model="test-model")
    result = agent.handle("edit the config file")

    assert result.answer == "Which file do you mean?"
    assert result.awaiting_user_input is True
    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "ask_user"
    # Only the first model call happened -- the loop never asked the model
    # for a second round after the question.
    assert mock_post.call_count == 1


def test_react_agent_stream_stops_immediately_on_ask_user():
    agent = ReActAgent(model="test-model")
    agent._call_model_stream = MagicMock(
        return_value=iter(
            [
                {
                    "message": {
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
                }
            ]
        )
    )

    events = list(agent.handle_stream("edit the config file"))

    final_events = [e for e in events if e["type"] == "final"]
    assert len(final_events) == 1
    result = final_events[0]["result"]
    assert result.answer == "Which file do you mean?"
    assert result.awaiting_user_input is True
    agent._call_model_stream.assert_called_once()


