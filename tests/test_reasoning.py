from unittest.mock import MagicMock, patch

from app.reasoning import _split_think_tags, generate_with_reasoning, stream_with_reasoning


def _mock_chat_response(message: dict):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": message}
    return mock_resp


def test_generate_with_reasoning_never_sends_tools():
    # Regression: live-testing deepseek-r1:14b with a real `tools` payload
    # showed it does NOT reliably use tool-calling despite Ollama listing
    # "tools" as a capability -- it either ignored a calculator tool and
    # miscalculated the answer itself, or narrated what a shell command would
    # hypothetically return instead of actually calling it (never emitting a
    # real tool_calls field). This module must stay pure chat-completion --
    # see the module docstring for the specific tests that proved this.
    import inspect

    source = inspect.getsource(generate_with_reasoning)
    assert '"tools"' not in source
    assert "bind_tools" not in source


def test_split_think_tags_extracts_reasoning_and_answer():
    content = "<think>step one, then step two</think>The final answer is 42."
    reasoning, answer = _split_think_tags(content)

    assert reasoning == "step one, then step two"
    assert answer == "The final answer is 42."


def test_split_think_tags_with_no_think_block_returns_content_unchanged():
    reasoning, answer = _split_think_tags("Just a plain answer.")

    assert reasoning == ""
    assert answer == "Just a plain answer."


@patch("app.reasoning.httpx.post")
def test_generate_with_reasoning_uses_native_thinking_field(mock_post):
    # Modern Ollama + reasoning-capable model: thinking/content already split.
    mock_post.return_value = _mock_chat_response(
        {
            "role": "assistant",
            "thinking": "First I consider X, then Y.",
            "content": "The answer is 42.",
        }
    )

    result = generate_with_reasoning("What is the meaning of life?", model="deepseek-r1:14b")

    assert result.reasoning == "First I consider X, then Y."
    assert result.answer == "The answer is 42."
    assert result.model == "deepseek-r1:14b"

    # Confirm the "think" flag was actually requested from Ollama.
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["think"] is True
    assert kwargs["json"]["model"] == "deepseek-r1:14b"


@patch("app.reasoning.httpx.post")
def test_generate_with_reasoning_falls_back_to_think_tags(mock_post):
    # Older Ollama / model inlines <think> instead of a separate field.
    mock_post.return_value = _mock_chat_response(
        {
            "role": "assistant",
            "content": "<think>Reasoning inline here.</think>Final answer here.",
        }
    )

    result = generate_with_reasoning("prompt")

    assert result.reasoning == "Reasoning inline here."
    assert result.answer == "Final answer here."


@patch("app.reasoning.httpx.post")
def test_generate_with_reasoning_defaults_to_configured_model(mock_post):
    mock_post.return_value = _mock_chat_response({"role": "assistant", "content": "ok"})

    generate_with_reasoning("prompt")

    _, kwargs = mock_post.call_args
    from app.config import OLLAMA_REASONING_MODEL

    assert kwargs["json"]["model"] == OLLAMA_REASONING_MODEL


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


@patch("app.reasoning.httpx.stream")
def test_stream_with_reasoning_yields_reasoning_then_answer_chunks(mock_stream):
    mock_stream.return_value = _mock_stream_context(
        [
            '{"message": {"thinking": "First I think about it."}}',
            '{"message": {"content": "The answer is "}}',
            '{"message": {"content": "42."}}',
        ]
    )

    chunks = list(stream_with_reasoning("What is 6*7?"))

    assert chunks == [
        ("reasoning", "First I think about it."),
        ("answer", "The answer is "),
        ("answer", "42."),
    ]


@patch("app.reasoning.httpx.stream")
def test_stream_with_reasoning_skips_blank_lines(mock_stream):
    # Ollama's NDJSON stream can include blank keep-alive lines between real
    # chunks -- these must be skipped, not passed to json.loads (which would
    # raise a JSONDecodeError and crash the whole stream).
    mock_stream.return_value = _mock_stream_context(
        ["", '{"message": {"content": "hi"}}', ""]
    )

    chunks = list(stream_with_reasoning("hello"))

    assert chunks == [("answer", "hi")]


@patch("app.reasoning.httpx.stream")
def test_stream_with_reasoning_requests_think_flag_and_configured_model(mock_stream):
    mock_stream.return_value = _mock_stream_context(['{"message": {"content": "ok"}}'])

    list(stream_with_reasoning("prompt", model="deepseek-r1:14b"))

    _, kwargs = mock_stream.call_args
    assert kwargs["json"]["think"] is True
    assert kwargs["json"]["model"] == "deepseek-r1:14b"
    assert kwargs["json"]["stream"] is True


