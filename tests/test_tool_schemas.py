from app.tool_schemas import validate_tool_args


def test_validate_tool_args_passes_through_valid_args():
    coerced, error = validate_tool_args("calculator", {"expression": "1 + 1"})

    assert error is None
    assert coerced == {"expression": "1 + 1"}


def test_validate_tool_args_coerces_string_boolean_to_real_boolean():
    # Regression test: a local tool-calling model was observed emitting
    # write_file's 'append' argument as the STRING "False" rather than a
    # real JSON boolean. pydantic's non-strict bool validation accepts
    # common string spellings ("false"/"False"/"0"/etc.) and coerces them
    # to the real bool -- so this should succeed with a real bool, not
    # reject the call or pass the string through unchanged.
    coerced, error = validate_tool_args(
        "write_file",
        {"file_path": "/tmp/x.txt", "text": "hello", "append": "False"},
    )

    assert error is None
    assert coerced["append"] is False
    assert isinstance(coerced["append"], bool)


def test_validate_tool_args_defaults_optional_fields():
    coerced, error = validate_tool_args("list_directory", {})

    assert error is None
    assert coerced == {"dir_path": "."}


def test_validate_tool_args_reports_missing_required_field():
    coerced, error = validate_tool_args("read_file", {})

    assert coerced is None
    assert error is not None
    assert "read_file" in error
    assert "file_path" in error


def test_validate_tool_args_reports_missing_required_field_with_other_fields_present():
    coerced, error = validate_tool_args("write_file", {"file_path": "/tmp/x.txt"})

    assert coerced is None
    assert "'text'" in error


def test_validate_tool_args_error_tells_model_to_retry_the_same_tool():
    # The whole point of validating client-side is so the error can be fed
    # back into the model's own context for a self-correcting second pass
    # -- the message must actually say to retry, not just report failure.
    _, error = validate_tool_args("read_file", {})

    assert "Re-call this SAME tool again" in error


def test_validate_tool_args_passes_through_unknown_tool_name_unvalidated():
    # A tool without a registered schema should never be rejected outright
    # -- it just skips this extra safety net.
    coerced, error = validate_tool_args("some_future_tool", {"anything": 1})

    assert error is None
    assert coerced == {"anything": 1}


def test_validate_tool_args_rejects_unexpected_extra_field_for_no_args_tool():
    coerced, error = validate_tool_args("system_resources", {"unexpected": "value"})

    assert coerced is None
    assert "system_resources" in error


