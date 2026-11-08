from unittest.mock import MagicMock, patch

from app.reasoning_react_agent import ReasoningReActAgent


def _mock_chat_response(message: dict):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": message}
    return mock_resp


@patch("app.reasoning_react_agent.httpx.post")
def test_answers_directly_with_final_answer_marker(mock_post):
    mock_post.return_value = _mock_chat_response(
        {
            "role": "assistant",
            "content": "Final Answer: The capital of France is Paris.",
            "thinking": "This is general knowledge, no tool needed.",
        }
    )

    agent = ReasoningReActAgent(model="test-model")
    result = agent.handle("What's the capital of France?")

    assert result.answer == "The capital of France is Paris."
    assert result.reasoning == "This is general knowledge, no tool needed."
    assert result.steps == []
    assert result.iterations == 1
    assert result.hit_iteration_limit is False


@patch("app.reasoning_react_agent.httpx.post")
def test_falls_back_to_raw_content_when_no_final_answer_marker(mock_post):
    # Regression: if the model answers without using either marker at all,
    # we should still return SOMETHING useful rather than an empty string.
    mock_post.return_value = _mock_chat_response(
        {"role": "assistant", "content": "It's Paris.", "thinking": ""}
    )

    agent = ReasoningReActAgent(model="test-model")
    result = agent.handle("What's the capital of France?")

    assert result.answer == "It's Paris."


@patch("app.reasoning_react_agent.httpx.post")
def test_calls_a_tool_then_answers(mock_post):
    mock_post.side_effect = [
        _mock_chat_response(
            {
                "role": "assistant",
                "content": 'Action: calculator\nAction Input: {"expression": "12 * 7"}',
                "thinking": "I should use the calculator for exact arithmetic.",
            }
        ),
        _mock_chat_response(
            {
                "role": "assistant",
                "content": "Final Answer: 12 * 7 is 84.",
                "thinking": "The observation confirms 84.",
            }
        ),
    ]

    agent = ReasoningReActAgent(model="test-model")
    result = agent.handle("What's 12 times 7?")

    assert result.answer == "12 * 7 is 84."
    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "calculator"
    assert result.steps[0].tool_args == {"expression": "12 * 7"}
    assert result.steps[0].observation == "84"
    assert result.steps[0].reasoning == "I should use the calculator for exact arithmetic."
    assert result.reasoning == "The observation confirms 84."
    assert result.iterations == 2


@patch("app.reasoning_react_agent.httpx.post")
def test_feeds_real_observation_back_as_a_user_turn(mock_post):
    mock_post.side_effect = [
        _mock_chat_response(
            {
                "role": "assistant",
                "content": 'Action: calculator\nAction Input: {"expression": "2 + 2"}',
                "thinking": "",
            }
        ),
        _mock_chat_response({"role": "assistant", "content": "Final Answer: 4", "thinking": ""}),
    ]

    agent = ReasoningReActAgent(model="test-model")
    agent.handle("what is 2+2")

    second_call_messages = mock_post.call_args_list[1].kwargs["json"]["messages"]
    assert second_call_messages[-1] == {"role": "user", "content": "Observation: 4"}


@patch("app.reasoning_react_agent.httpx.post")
def test_handles_malformed_action_input_gracefully(mock_post):
    mock_post.side_effect = [
        _mock_chat_response(
            {
                "role": "assistant",
                "content": "Action: calculator\nAction Input: {not valid json",
                "thinking": "",
            }
        ),
        _mock_chat_response(
            {"role": "assistant", "content": "Final Answer: couldn't compute that.", "thinking": ""}
        ),
    ]

    agent = ReasoningReActAgent(model="test-model")
    result = agent.handle("compute something weird")

    # Malformed JSON args -- our regex requires a balanced-looking `{...}`
    # with no unmatched braces, so this simply never matches as an action at
    # all and the whole reply is treated as the (non-marker) answer instead.
    assert result.steps == []


@patch("app.reasoning_react_agent.httpx.post")
def test_handles_unknown_tool_name_gracefully(mock_post):
    mock_post.side_effect = [
        _mock_chat_response(
            {
                "role": "assistant",
                "content": "Action: nonexistent_tool\nAction Input: {}",
                "thinking": "",
            }
        ),
        _mock_chat_response(
            {"role": "assistant", "content": "Final Answer: Sorry, I couldn't do that.", "thinking": ""}
        ),
    ]

    agent = ReasoningReActAgent(model="test-model")
    result = agent.handle("do something weird")

    assert "unknown tool" in result.steps[0].observation


@patch("app.reasoning_react_agent.httpx.post")
def test_stops_at_max_iterations(mock_post):
    looping_response = _mock_chat_response(
        {
            "role": "assistant",
            "content": 'Action: calculator\nAction Input: {"expression": "1+1"}',
            "thinking": "",
        }
    )
    mock_post.return_value = looping_response

    agent = ReasoningReActAgent(model="test-model")
    result = agent.handle("keep going forever")

    assert result.hit_iteration_limit is True
    assert len(result.steps) == 5  # MAX_ITERATIONS


@patch("app.reasoning_react_agent.httpx.post")
def test_requests_think_flag_and_stop_sequences(mock_post):
    mock_post.return_value = _mock_chat_response(
        {"role": "assistant", "content": "Final Answer: hi", "thinking": ""}
    )

    agent = ReasoningReActAgent(model="deepseek-r1:14b")
    agent.handle("hello")

    sent = mock_post.call_args.kwargs["json"]
    assert sent["think"] is True
    assert sent["model"] == "deepseek-r1:14b"
    assert "Observation" in sent["options"]["stop"]


@patch("app.reasoning_react_agent.httpx.post")
def test_includes_prior_history_in_the_model_request(mock_post):
    mock_post.return_value = _mock_chat_response(
        {"role": "assistant", "content": "Final Answer: the root dir is '/'.", "thinking": ""}
    )

    agent = ReasoningReActAgent(model="test-model")
    history = [
        {"role": "user", "content": "What is the current working directory?"},
        {"role": "assistant", "content": "/Users/emil/project"},
    ]
    agent.handle("and what is the root directory?", history=history)

    sent_messages = mock_post.call_args.kwargs["json"]["messages"]
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[1] == history[0]
    assert sent_messages[2] == history[1]
    assert sent_messages[3] == {"role": "user", "content": "and what is the root directory?"}

