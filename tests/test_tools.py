from app.tools import calculator, knowledge_base_lookup, run_shell_command


def test_calculator_tool_basic_arithmetic():
    assert calculator.invoke({"expression": "2 + 2"}) == "4"


def test_calculator_tool_handles_precedence():
    assert calculator.invoke({"expression": "2 + 3 * 4"}) == "14"


def test_calculator_tool_rejects_unsafe_expression():
    assert calculator.invoke({"expression": "__import__('os').system('ls')"}) == (
        "Error: could not evaluate expression"
    )


def test_knowledge_base_lookup_known_topic():
    result = knowledge_base_lookup.invoke({"topic": "refund policy"})
    assert "5-7 business days" in result


def test_knowledge_base_lookup_unknown_topic():
    result = knowledge_base_lookup.invoke({"topic": "unknown thing"})
    assert "No internal knowledge found" in result


def test_run_shell_command_allows_pwd():
    result = run_shell_command.invoke({"command": "pwd"})
    assert not result.startswith("Error")
    assert "/" in result


def test_run_shell_command_allows_echo_with_args():
    result = run_shell_command.invoke({"command": "echo hello world"})
    assert result == "hello world"


def test_run_shell_command_rejects_empty_command():
    result = run_shell_command.invoke({"command": "   "})
    assert result == "Error: empty command"


def test_run_shell_command_supports_pipes():
    # Unlike the old allowlisted version, this now runs through a real shell
    # (shell=True), so pipes/redirects/chaining all work as normal bash.
    result = run_shell_command.invoke({"command": "echo hello | wc -l"})
    assert result.strip() == "1"


def test_run_shell_command_supports_arbitrary_programs():
    # No allowlist anymore -- any program on PATH can be run, not just a
    # fixed safe set.
    result = run_shell_command.invoke({"command": "python3 -c 'print(2 + 2)'"})
    assert result.strip() == "4"


def test_run_shell_command_reports_nonzero_exit_code():
    result = run_shell_command.invoke({"command": "ls /path/does/not/exist"})
    assert result.startswith("[exit code")


def test_run_shell_command_docstring_mentions_full_freedom():
    # The docstring is literally the only information the LLM has about this
    # tool's capabilities -- it must reflect that pipes/chaining/any program
    # are allowed now, not the old restricted allowlist behavior.
    docstring = run_shell_command.description
    assert "pipes" in docstring.lower()
    assert "no allowlist" in docstring.lower() or "no restrictions" in docstring.lower()
