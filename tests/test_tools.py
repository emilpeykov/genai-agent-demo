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


def test_run_shell_command_rejects_pipe_with_clear_message():
    # Regression test: a real LLM once tried "ls -l | grep ^- | wc -l".
    # Since we run with shell=False, "|" was previously treated as a
    # literal (meaningless) argument, producing confusing per-token
    # "No such file or directory" errors instead of one clear message.
    result = run_shell_command.invoke({"command": "ls -l | grep ^- | wc -l"})
    assert result.startswith("Error: '|' is not supported")
    assert "No such file or directory" not in result


def test_run_shell_command_rejects_command_chaining():
    result = run_shell_command.invoke({"command": "pwd; rm -rf /"})
    assert result.startswith("Error: ';' is not supported")


def test_run_shell_command_rejects_command_substitution():
    result = run_shell_command.invoke({"command": "echo $(whoami)"})
    assert result.startswith("Error: '$(' is not supported")


def test_run_shell_command_docstring_warns_against_pipe_idiom():
    # Regression guard: a real LLM repeatedly reached for "ls | wc -l" style
    # pipelines when asked to "count files", since that's the standard Unix
    # idiom it saw during training. The tool's docstring (which is sent to
    # the model as its "description") must keep proactively warning against
    # this, or that failure mode comes back. See tools like 'wc'/'grep' being
    # explicitly called out as unavailable, with guidance to count manually.
    docstring = run_shell_command.description
    assert "wc" in docstring
    assert "count" in docstring.lower()
