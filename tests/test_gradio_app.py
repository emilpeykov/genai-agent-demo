from app.gradio_app import _clean_history_for_model


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
