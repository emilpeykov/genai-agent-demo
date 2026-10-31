from unittest.mock import MagicMock, patch

from app.react_agent import ReActAgent


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
    assert len(result.steps) == 5  # MAX_ITERATIONS


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
