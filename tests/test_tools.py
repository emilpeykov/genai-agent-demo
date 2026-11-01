from app.tools import (
    READ_FILE_MAX_CHARS,
    calculator,
    knowledge_base_lookup,
    read_file,
    run_shell_command,
    wikipedia_search,
)


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


def test_read_file_returns_exact_file_content(tmp_path):
    file_path = tmp_path / "README.md"
    file_path.write_text("# My Project\n\nThis project does X.\n")

    result = read_file.invoke({"path": str(file_path)})

    assert result == "# My Project\n\nThis project does X.\n"


def test_read_file_rejects_missing_path():
    result = read_file.invoke({"path": "/path/does/not/exist.md"})
    assert result == "Error: path '/path/does/not/exist.md' does not exist"


def test_read_file_rejects_a_directory(tmp_path):
    result = read_file.invoke({"path": str(tmp_path)})
    assert "is not a file" in result


def test_read_file_truncates_very_long_files(tmp_path):
    file_path = tmp_path / "big.txt"
    file_path.write_text("x" * (READ_FILE_MAX_CHARS + 500))

    result = read_file.invoke({"path": str(file_path)})

    assert result.endswith("... [truncated]")
    assert len(result) <= READ_FILE_MAX_CHARS + len("\n... [truncated]")


def test_read_file_rejects_binary_content(tmp_path):
    file_path = tmp_path / "image.png"
    file_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x01\x02")

    result = read_file.invoke({"path": str(file_path)})

    assert result.startswith("Error:")


def test_wikipedia_search_returns_summary_on_success(monkeypatch):
    import app.tools as tools_module

    class _FakeWikipedia:
        def run(self, query: str) -> str:
            return f"Summary about {query}"

    monkeypatch.setattr(tools_module, "_wikipedia", _FakeWikipedia())

    result = wikipedia_search.invoke({"query": "Ada Lovelace"})

    assert result == "Summary about Ada Lovelace"


def test_wikipedia_search_handles_no_results(monkeypatch):
    import app.tools as tools_module

    class _FakeWikipedia:
        def run(self, query: str) -> str:
            return ""

    monkeypatch.setattr(tools_module, "_wikipedia", _FakeWikipedia())

    result = wikipedia_search.invoke({"query": "asdkjfhaslkdjfh"})

    assert "No Wikipedia results found" in result


def test_wikipedia_search_handles_network_errors_gracefully(monkeypatch):
    import app.tools as tools_module

    class _FakeWikipedia:
        def run(self, query: str) -> str:
            raise ConnectionError("no network")

    monkeypatch.setattr(tools_module, "_wikipedia", _FakeWikipedia())

    result = wikipedia_search.invoke({"query": "anything"})

    assert result.startswith("Error searching Wikipedia")
    assert "no network" in result
