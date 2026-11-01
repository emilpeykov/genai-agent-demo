from app.tools import (
    calculator,
    copy_file,
    edit_file,
    file_delete,
    file_search,
    knowledge_base_lookup,
    list_directory,
    move_file,
    read_file,
    run_shell_command,
    wikipedia_search,
    write_file,
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

    result = read_file.invoke({"file_path": str(file_path)})

    assert result == "# My Project\n\nThis project does X.\n"


def test_read_file_rejects_missing_path():
    result = read_file.invoke({"file_path": "/path/does/not/exist.md"})
    assert "no such file or directory" in result


def test_read_file_rejects_a_directory(tmp_path):
    result = read_file.invoke({"file_path": str(tmp_path)})
    assert result.startswith("Error:")


def test_read_file_rejects_binary_content(tmp_path):
    file_path = tmp_path / "image.png"
    file_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x01\x02")

    result = read_file.invoke({"file_path": str(file_path)})

    assert result.startswith("Error:")


def test_write_file_creates_new_file(tmp_path):
    file_path = tmp_path / "new.txt"

    result = write_file.invoke({"file_path": str(file_path), "text": "hello world"})

    assert "successfully" in result
    assert file_path.read_text() == "hello world"


def test_write_file_overwrites_existing_file_by_default(tmp_path):
    file_path = tmp_path / "existing.txt"
    file_path.write_text("old content")

    write_file.invoke({"file_path": str(file_path), "text": "new content"})

    assert file_path.read_text() == "new content"


def test_write_file_appends_when_append_is_true(tmp_path):
    file_path = tmp_path / "log.txt"
    file_path.write_text("line 1\n")

    result = write_file.invoke(
        {"file_path": str(file_path), "text": "line 2\n", "append": True}
    )

    assert "successfully" in result
    assert file_path.read_text() == "line 1\nline 2\n"


def test_write_file_creates_one_missing_parent_directory(tmp_path):
    file_path = tmp_path / "subdir" / "new.txt"

    result = write_file.invoke({"file_path": str(file_path), "text": "nested content"})

    assert "successfully" in result
    assert file_path.read_text() == "nested content"


def test_write_file_reports_error_for_deeply_missing_parent_directory(tmp_path):
    # write_file's parent-dir creation is non-recursive (mkdir(parents=False)),
    # so more than one missing directory level still fails -- unlike the old
    # hand-written write_file, which created arbitrarily deep parent paths.
    file_path = tmp_path / "subdir" / "nested" / "new.txt"

    result = write_file.invoke({"file_path": str(file_path), "text": "nested content"})

    assert result.startswith("Error:")
    assert not file_path.exists()


def test_list_directory_lists_entries(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()

    result = list_directory.invoke({"dir_path": str(tmp_path)})

    assert "a.txt" in result
    assert "sub" in result


def test_list_directory_reports_empty_directory(tmp_path):
    result = list_directory.invoke({"dir_path": str(tmp_path)})
    assert "No files found" in result


def test_edit_file_replaces_unique_match(tmp_path):
    file_path = tmp_path / "f.py"
    file_path.write_text("def foo():\n    return 1\n")

    result = edit_file.invoke(
        {"path": str(file_path), "old_text": "return 1", "new_text": "return 2"}
    )

    assert "replaced 1 occurrence" in result
    assert file_path.read_text() == "def foo():\n    return 2\n"


def test_edit_file_rejects_missing_old_text(tmp_path):
    file_path = tmp_path / "f.py"
    file_path.write_text("def foo():\n    return 1\n")

    result = edit_file.invoke(
        {"path": str(file_path), "old_text": "does not exist", "new_text": "x"}
    )

    assert "not found" in result
    assert file_path.read_text() == "def foo():\n    return 1\n"


def test_edit_file_rejects_ambiguous_match(tmp_path):
    file_path = tmp_path / "f.py"
    file_path.write_text("x = 1\nx = 1\n")

    result = edit_file.invoke({"path": str(file_path), "old_text": "x = 1", "new_text": "x = 2"})

    assert "matches 2 locations" in result
    assert file_path.read_text() == "x = 1\nx = 1\n"


def test_edit_file_rejects_missing_file():
    result = edit_file.invoke(
        {"path": "/path/does/not/exist.py", "old_text": "a", "new_text": "b"}
    )

    assert result == "Error: path '/path/does/not/exist.py' does not exist"


def test_wikipedia_search_returns_summary_on_success(monkeypatch):
    import app.tools as tools_module

    class _FakeWikipediaTool:
        def invoke(self, args: dict) -> str:
            return f"Summary about {args['query']}"

    monkeypatch.setattr(tools_module, "_wikipedia_tool", _FakeWikipediaTool())

    result = wikipedia_search.invoke({"query": "Ada Lovelace"})

    assert result == "Summary about Ada Lovelace"


def test_wikipedia_search_handles_no_results(monkeypatch):
    import app.tools as tools_module

    class _FakeWikipediaTool:
        def invoke(self, args: dict) -> str:
            return ""

    monkeypatch.setattr(tools_module, "_wikipedia_tool", _FakeWikipediaTool())

    result = wikipedia_search.invoke({"query": "asdkjfhaslkdjfh"})

    assert "No Wikipedia results found" in result


def test_wikipedia_search_handles_network_errors_gracefully(monkeypatch):
    import app.tools as tools_module

    class _FakeWikipediaTool:
        def invoke(self, args: dict) -> str:
            raise ConnectionError("no network")

    monkeypatch.setattr(tools_module, "_wikipedia_tool", _FakeWikipediaTool())

    result = wikipedia_search.invoke({"query": "anything"})

    assert result.startswith("Error searching Wikipedia")
    assert "no network" in result


def test_file_search_finds_matching_files(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.py").write_text("b")

    result = file_search.invoke({"dir_path": str(tmp_path), "pattern": "*.txt"})

    assert "a.txt" in result
    assert "b.py" not in result


def test_copy_file_duplicates_content(tmp_path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("hello")

    copy_file.invoke({"source_path": str(src), "destination_path": str(dst)})

    assert dst.read_text() == "hello"
    assert src.read_text() == "hello"


def test_move_file_relocates_content(tmp_path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("hello")

    move_file.invoke({"source_path": str(src), "destination_path": str(dst)})

    assert dst.read_text() == "hello"
    assert not src.exists()


def test_file_delete_removes_file(tmp_path):
    target = tmp_path / "doomed.txt"
    target.write_text("bye")

    result = file_delete.invoke({"file_path": str(target)})

    assert not target.exists()
    assert "deleted" in result.lower()
