from unittest.mock import MagicMock, patch

from app.reasoning_langgraph_agent import ReasoningLangGraphAgent


def _mock_chat_response(message: dict):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": message}
    return mock_resp


@patch("app.reasoning_langgraph_agent.httpx.post")
def test_answers_directly_with_final_answer_marker(mock_post):
    mock_post.return_value = _mock_chat_response(
        {
            "role": "assistant",
            "content": "Final Answer: The capital of France is Paris.",
            "thinking": "General knowledge, no tool needed.",
        }
    )

    agent = ReasoningLangGraphAgent(model="test-model")
    result = agent.handle("What's the capital of France?")

    assert result.answer == "The capital of France is Paris."
    assert result.reasoning == "General knowledge, no tool needed."
    assert result.steps == []
    assert result.iterations == 1
    assert result.hit_iteration_limit is False


@patch("app.reasoning_langgraph_agent.httpx.post")
def test_calls_a_tool_then_answers(mock_post):
    mock_post.side_effect = [
        _mock_chat_response(
            {
                "role": "assistant",
                "content": 'Action: calculator\nAction Input: {"expression": "12 * 7"}',
                "thinking": "Use the calculator for exact arithmetic.",
            }
        ),
        _mock_chat_response(
            {"role": "assistant", "content": "Final Answer: 12 * 7 is 84.", "thinking": "Confirmed."}
        ),
    ]

    agent = ReasoningLangGraphAgent(model="test-model")
    result = agent.handle("What's 12 times 7?")

    assert result.answer == "12 * 7 is 84."
    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "calculator"
    assert result.steps[0].tool_args == {"expression": "12 * 7"}
    assert result.steps[0].observation == "84"
    assert result.iterations == 2


@patch("app.reasoning_langgraph_agent.httpx.post")
def test_self_correction_edge_feeds_real_json_error_back_to_model(mock_post):
    # The behavior the user specifically asked for: a broken tool call
    # (invalid JSON in Action Input) must not crash -- a conditional edge
    # routes back to the agent node with the REAL parse error as feedback,
    # and the model gets a chance to retry, exactly like a human reviewing
    # a stack trace.
    mock_post.side_effect = [
        _mock_chat_response(
            {
                "role": "assistant",
                "content": "Action: calculator\nAction Input: {expression: 12 * 7}",  # invalid JSON (unquoted key)
                "thinking": "",
            }
        ),
        _mock_chat_response(
            {
                "role": "assistant",
                "content": 'Action: calculator\nAction Input: {"expression": "12 * 7"}',
                "thinking": "",
            }
        ),
        _mock_chat_response(
            {"role": "assistant", "content": "Final Answer: 84.", "thinking": ""}
        ),
    ]

    agent = ReasoningLangGraphAgent(model="test-model")
    result = agent.handle("What's 12 times 7?")

    # No tool was actually invoked on the first (broken) attempt -- only
    # the second, valid attempt shows up as a real step.
    assert len(result.steps) == 1
    assert result.steps[0].observation == "84"
    assert result.answer == "84."
    assert result.iterations == 3

    # The self-correction feedback message sent back to the model on the
    # SECOND call must contain the real JSON error, not a generic message.
    second_call_messages = mock_post.call_args_list[1].kwargs["json"]["messages"]
    feedback = second_call_messages[-1]["content"]
    assert "not valid JSON" in feedback
    assert "Action: calculator" in feedback


@patch("app.reasoning_langgraph_agent.httpx.post")
def test_missing_action_input_for_multi_arg_tool_gets_json_skeleton_retry(mock_post):
    mock_post.side_effect = [
        _mock_chat_response(
            {"role": "assistant", "content": "Action: edit_file", "thinking": ""}
        ),
        _mock_chat_response(
            {"role": "assistant", "content": "Final Answer: gave up.", "thinking": ""}
        ),
    ]

    agent = ReasoningLangGraphAgent(model="test-model")
    agent.handle("edit a file")

    second_call_messages = mock_post.call_args_list[1].kwargs["json"]["messages"]
    feedback = second_call_messages[-1]["content"]
    assert "path" in feedback
    assert "old_text" in feedback
    assert "new_text" in feedback
    assert "{" in feedback and "}" in feedback


@patch("app.reasoning_langgraph_agent.httpx.post")
def test_handles_zero_argument_tool_with_no_action_input_line(mock_post):
    mock_post.side_effect = [
        _mock_chat_response(
            {"role": "assistant", "content": "Action: system_resources", "thinking": ""}
        ),
        _mock_chat_response(
            {"role": "assistant", "content": "Final Answer: memory reported.", "thinking": ""}
        ),
    ]

    agent = ReasoningLangGraphAgent(model="test-model")
    result = agent.handle("How much memory is this machine using?")

    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "system_resources"
    assert result.steps[0].tool_args == {}
    assert result.answer == "memory reported."


@patch("app.reasoning_langgraph_agent.httpx.post")
def test_stops_at_max_iterations(mock_post):
    looping_response = _mock_chat_response(
        {
            "role": "assistant",
            "content": 'Action: calculator\nAction Input: {"expression": "1+1"}',
            "thinking": "",
        }
    )
    mock_post.return_value = looping_response

    agent = ReasoningLangGraphAgent(model="test-model")
    result = agent.handle("keep going forever")

    assert result.hit_iteration_limit is True
    assert len(result.steps) == 5  # MAX_ITERATIONS


@patch("app.reasoning_langgraph_agent.httpx.post")
def test_requests_think_flag_and_stop_sequences(mock_post):
    mock_post.return_value = _mock_chat_response(
        {"role": "assistant", "content": "Final Answer: hi", "thinking": ""}
    )

    agent = ReasoningLangGraphAgent(model="deepseek-r1:14b")
    agent.handle("hello")

    sent = mock_post.call_args.kwargs["json"]
    assert sent["think"] is True
    assert sent["model"] == "deepseek-r1:14b"
    assert "Observation" in sent["options"]["stop"]


@patch("app.reasoning_langgraph_agent.httpx.post")
def test_includes_prior_history_in_the_model_request(mock_post):
    mock_post.return_value = _mock_chat_response(
        {"role": "assistant", "content": "Final Answer: the root dir is '/'.", "thinking": ""}
    )

    agent = ReasoningLangGraphAgent(model="test-model")
    history = [
        {"role": "user", "content": "What is the current working directory?"},
        {"role": "assistant", "content": "/Users/emil/project"},
    ]
    agent.handle("What about the root directory?", history=history)

    sent_messages = mock_post.call_args.kwargs["json"]["messages"]
    assert history[0] in sent_messages
    assert history[1] in sent_messages


@patch("app.reasoning_langgraph_agent.httpx.post")
def test_thread_id_remembers_conversation_without_caller_supplied_history(mock_post):
    # The checkpointer-backed path: no 'history' is ever passed in by the
    # caller, yet the SECOND call (same thread_id) must still see the FIRST
    # call's user prompt and assistant answer in the messages sent to the
    # model -- the server, not the caller, is responsible for continuity.
    mock_post.side_effect = [
        _mock_chat_response(
            {"role": "assistant", "content": "Final Answer: $2,528.27", "thinking": ""}
        ),
        _mock_chat_response(
            {"role": "assistant", "content": "Final Answer: $2,271.16", "thinking": ""}
        ),
    ]

    agent = ReasoningLangGraphAgent(model="test-model")
    agent.handle(
        "What's my monthly cost for a $400,000 loan at 6.5% for 30 years?",
        thread_id="user_session_123",
    )
    agent.handle(
        "What would it be if the rate dropped to 5.5% instead?",
        thread_id="user_session_123",
    )

    second_call_messages = mock_post.call_args_list[1].kwargs["json"]["messages"]
    contents = [m["content"] for m in second_call_messages]
    assert "What's my monthly cost for a $400,000 loan at 6.5% for 30 years?" in contents
    assert "$2,528.27" in contents
    assert "What would it be if the rate dropped to 5.5% instead?" in contents


@patch("app.reasoning_langgraph_agent.httpx.post")
def test_different_thread_ids_are_isolated(mock_post):
    mock_post.side_effect = [
        _mock_chat_response(
            {"role": "assistant", "content": "Final Answer: first thread answer", "thinking": ""}
        ),
        _mock_chat_response(
            {"role": "assistant", "content": "Final Answer: second thread answer", "thinking": ""}
        ),
    ]

    agent = ReasoningLangGraphAgent(model="test-model")
    agent.handle("Remember the number 42.", thread_id="thread-a")
    agent.handle("What number did I just mention?", thread_id="thread-b")

    second_call_messages = mock_post.call_args_list[1].kwargs["json"]["messages"]
    contents = [m["content"] for m in second_call_messages]
    assert "Remember the number 42." not in contents
    assert "What number did I just mention?" in contents


@patch("app.reasoning_langgraph_agent.httpx.post")
def test_stops_immediately_on_ask_user(mock_post):
    # 'ask_user' must short-circuit the graph right from inside the tool
    # node -- the question becomes this turn's "answer" and
    # awaiting_user_input is set, instead of routing back to the agent node
    # for another round it has no real reply to give yet.
    mock_post.side_effect = [
        _mock_chat_response(
            {
                "role": "assistant",
                "content": 'Action: ask_user\nAction Input: {"question": "Which file do you mean?"}',
                "thinking": "I need to clarify which file before proceeding.",
            }
        ),
    ]

    agent = ReasoningLangGraphAgent(model="test-model")
    result = agent.handle("edit the config file", thread_id="ask-user-thread")

    assert result.answer == "Which file do you mean?"
    assert result.awaiting_user_input is True
    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "ask_user"
    assert mock_post.call_count == 1


