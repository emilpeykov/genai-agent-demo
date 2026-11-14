from unittest.mock import patch

from app.gradio_app import _clean_history_for_model, _react_respond


def test_clean_history_strips_debug_metadata_footer_from_assistant_turns():
    history = [
        {"role": "user", "content": "What's the refund policy?"},
        {
            "role": "assistant",
            "content": "Refunds take 5-7 days.\n\n---\n*model: `llama3.1:8b` | iterations: 1*",
        },
    ]

    cleaned = _clean_history_for_model(history)

    assert cleaned == [
        {"role": "user", "content": "What's the refund policy?"},
        {"role": "assistant", "content": "Refunds take 5-7 days."},
    ]


def test_clean_history_ignores_non_message_entries():
    history = [
        {"role": "user", "content": "hello"},
        {"role": "system", "content": "should be ignored"},
        {"metadata": {"something": "else"}},
    ]

    cleaned = _clean_history_for_model(history)

    assert cleaned == [{"role": "user", "content": "hello"}]


def test_clean_history_handles_empty_list():
    assert _clean_history_for_model([]) == []


def test_react_respond_passes_thread_id_through_to_handle_stream():
    # Regression test: the ReAct tab's memory is now carried entirely by
    # thread_id (a per-tab gr.State) -- _react_respond must forward it
    # untouched to ReActLangGraphAgent.handle_stream and must NOT reconstruct
    # conversation_history/known_files from answer_history the old way.
    captured = {}

    def _fake_handle_stream(message, thread_id=None):
        captured["message"] = message
        captured["thread_id"] = thread_id
        yield {"type": "content_delta", "text": "hi there"}
        from app.react_agent import ReActResult

        yield {"type": "final", "result": ReActResult(answer="hi there", model="qwen3:14b")}

    with patch("app.gradio_app._react_agent") as mock_agent:
        mock_agent.handle_stream.side_effect = _fake_handle_stream
        events = list(_react_respond("hello", [], [], "tab-thread-xyz"))

    assert captured["message"] == "hello"
    assert captured["thread_id"] == "tab-thread-xyz"
    # Last yielded tuple preserves the SAME thread_id (unchanged across the
    # turn, unlike the old known_files list which grew every turn).
    assert events[-1][3] == "tab-thread-xyz"
    assert events[-1][1][-1]["content"] == "hi there"


def test_react_respond_shows_ask_user_question_in_answer_panel_not_blank():
    # Regression test: when 'ask_user' fires, the agent's loop stops BEFORE
    # any content_delta is ever streamed (the question only exists as
    # result.answer, see ReActLangGraphAgent.handle_stream's ask_user
    # short-circuit) -- naively falling back to content_acc (still "") left
    # the right-hand "Answer" panel blank, making the whole UI look stuck
    # with no indication a reply was expected. The question must be shown
    # there explicitly instead.
    def _fake_handle_stream(message, thread_id=None):
        from app.react_agent import ReActResult, ReActStep

        yield {
            "type": "tool_call",
            "tool_name": "ask_user",
            "tool_args": {"question": "Which file do you mean?"},
        }
        yield {
            "type": "observation",
            "tool_name": "ask_user",
            "observation": "(Question sent to the user -- awaiting their reply: Which file do you mean?)",
        }
        yield {
            "type": "final",
            "result": ReActResult(
                answer="Which file do you mean?",
                steps=[
                    ReActStep(
                        tool_name="ask_user",
                        tool_args={"question": "Which file do you mean?"},
                        observation="(Question sent to the user -- awaiting their reply: Which file do you mean?)",
                    )
                ],
                model="qwen3:14b",
                awaiting_user_input=True,
            ),
        }

    with patch("app.gradio_app._react_agent") as mock_agent:
        mock_agent.handle_stream.side_effect = _fake_handle_stream
        events = list(_react_respond("delete a file", [], [], "tab-thread-ask"))

    final_answer_content = events[-1][1][-1]["content"]
    assert "Which file do you mean?" in final_answer_content
    assert final_answer_content.strip() != ""


