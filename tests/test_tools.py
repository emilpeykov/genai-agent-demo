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


def test_run_shell_command_rejects_disallowed_program():
    result = run_shell_command.invoke({"command": "rm -rf /"})
    assert result.startswith("Error: 'rm' is not an allowed command")


def test_run_shell_command_rejects_unparseable_input():
    result = run_shell_command.invoke({"command": "echo 'unterminated"})
    assert result.startswith("Error: could not parse command")


def test_run_shell_command_rejects_empty_command():
    result = run_shell_command.invoke({"command": "   "})
    assert result == "Error: empty command"
