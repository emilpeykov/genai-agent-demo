from app.tools import (
    arxiv_search,
    ask_user,
    calculator,
    copy_file,
    crawl_website,
    edit_file,
    fetch_webpage,
    file_delete,
    file_search,
    get_errors,
    google_search,
    grep_search,
    knowledge_base_lookup,
    list_directory,
    move_file,
    read_file,
    run_shell_command,
    stackexchange_search,
    web_search,
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


def test_run_shell_command_timeout_is_long_enough_for_this_apps_own_endpoints():
    # Regression test: the timeout used to be 10s, which is shorter than
    # this app's OWN slower LLM-backed endpoints (e.g. curling
    # http://127.0.0.1:8000/chat can legitimately take well over 10s with a
    # "powerful" tier model) can take to respond -- every such self-curl was
    # killed by the tool's own timeout before the real answer came back,
    # producing a generic "command timed out" observation that the model
    # then misread as "the server is down". 90s gives real headroom.
    from app.tools import SHELL_COMMAND_TIMEOUT_SECONDS

    assert SHELL_COMMAND_TIMEOUT_SECONDS >= 60


def test_run_shell_command_reports_a_real_timeout(monkeypatch):
    # Patch the timeout down to 1s so this test doesn't actually take 90s.
    monkeypatch.setattr("app.tools.SHELL_COMMAND_TIMEOUT_SECONDS", 1)
    result = run_shell_command.invoke({"command": "sleep 5"})
    assert result == "Error: command timed out after 1s"


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


def test_web_search_returns_summary_on_success(monkeypatch):
    import app.tools as tools_module

    class _FakeDdgWrapper:
        def run(self, query: str) -> str:
            return f"Web summary about {query}"

    monkeypatch.setattr(tools_module, "_ddg_wrapper", _FakeDdgWrapper())

    result = web_search.invoke({"query": "OpenKBS infrastructure"})

    assert result == "Web summary about OpenKBS infrastructure"


def test_web_search_handles_no_results(monkeypatch):
    import app.tools as tools_module

    class _FakeDdgWrapper:
        def run(self, query: str) -> str:
            return ""

    monkeypatch.setattr(tools_module, "_ddg_wrapper", _FakeDdgWrapper())

    result = web_search.invoke({"query": "asdkjfhaslkdjfh"})

    assert "No web search results found" in result


def test_web_search_handles_network_errors_gracefully(monkeypatch):
    import app.tools as tools_module

    class _FakeDdgWrapper:
        def run(self, query: str) -> str:
            raise ConnectionError("no network")

    monkeypatch.setattr(tools_module, "_ddg_wrapper", _FakeDdgWrapper())

    result = web_search.invoke({"query": "anything"})

    assert result.startswith("Error searching the web")
    assert "no network" in result


def test_web_search_reports_missing_dependency_clearly(monkeypatch):
    import app.tools as tools_module

    monkeypatch.setattr(tools_module, "_ddg_wrapper", None)
    monkeypatch.setattr(tools_module, "_ddg_import_error", ImportError("no module"))

    result = web_search.invoke({"query": "anything"})

    assert result.startswith("Error: web_search is unavailable")


def test_google_search_returns_summary_on_success(monkeypatch):
    import app.tools as tools_module

    class _FakeGoogleWrapper:
        def run(self, query: str) -> str:
            return f"Google summary about {query}"

    monkeypatch.setattr(tools_module, "_google_wrapper", _FakeGoogleWrapper())

    result = google_search.invoke({"query": "OpenKBS infrastructure"})

    assert result == "Google summary about OpenKBS infrastructure"


def test_google_search_handles_network_errors_gracefully(monkeypatch):
    import app.tools as tools_module

    class _FakeGoogleWrapper:
        def run(self, query: str) -> str:
            raise ConnectionError("no network")

    monkeypatch.setattr(tools_module, "_google_wrapper", _FakeGoogleWrapper())

    result = google_search.invoke({"query": "anything"})

    assert result.startswith("Error searching Google")
    assert "no network" in result


def test_google_search_falls_back_to_bing_scrape_without_api_key(monkeypatch):
    # No SERPER_API_KEY configured -- 'google_search' must still return real
    # results via the key-free Bing HTML scrape instead of just reporting
    # itself unavailable.
    import app.tools as tools_module

    def _fake_scrape(query, num_results=5):
        return [
            {
                "title": "OpenKBS",
                "url": "https://openkbs.com",
                "description": "An agentic services platform.",
            }
        ]

    monkeypatch.setattr(tools_module, "_google_wrapper", None)
    monkeypatch.setattr(tools_module, "_bing_search_scrape", _fake_scrape)

    result = google_search.invoke({"query": "OpenKBS agentic services"})

    assert "OpenKBS" in result
    assert "https://openkbs.com" in result
    assert "An agentic services platform." in result


def test_google_search_scrape_fallback_handles_errors_gracefully(monkeypatch):
    import app.tools as tools_module

    def _fake_scrape(query, num_results=5):
        raise ConnectionError("blocked")

    monkeypatch.setattr(tools_module, "_google_wrapper", None)
    monkeypatch.setattr(tools_module, "_bing_search_scrape", _fake_scrape)

    result = google_search.invoke({"query": "anything"})

    assert result.startswith("Error searching for")
    assert "blocked" in result


def test_google_search_scrape_fallback_reports_no_results(monkeypatch):
    import app.tools as tools_module

    monkeypatch.setattr(tools_module, "_google_wrapper", None)
    monkeypatch.setattr(tools_module, "_bing_search_scrape", lambda query, **kw: [])

    result = google_search.invoke({"query": "anything"})

    assert result == "No search results found for 'anything'."


def test_decode_bing_redirect_url_extracts_real_target_url():
    from app.tools import _decode_bing_redirect_url

    href = (
        "https://www.bing.com/ck/a?!&&p=abc123&u=a1aHR0cHM6Ly9vcGVuYWkuY29tLw&ntb=1"
    )

    assert _decode_bing_redirect_url(href) == "https://openai.com/"


def test_decode_bing_redirect_url_passes_through_non_matching_href():
    from app.tools import _decode_bing_redirect_url

    assert _decode_bing_redirect_url("https://example.com") == "https://example.com"



def test_arxiv_search_returns_summary_on_success(monkeypatch):
    import app.tools as tools_module

    class _FakeArxivTool:
        def invoke(self, args: dict) -> str:
            return f"Arxiv summary about {args['query']}"

    monkeypatch.setattr(tools_module, "_arxiv_tool", _FakeArxivTool())

    result = arxiv_search.invoke({"query": "attention is all you need"})

    assert result == "Arxiv summary about attention is all you need"


def test_arxiv_search_handles_network_errors_gracefully(monkeypatch):
    import app.tools as tools_module

    class _FakeArxivTool:
        def invoke(self, args: dict) -> str:
            raise ConnectionError("no network")

    monkeypatch.setattr(tools_module, "_arxiv_tool", _FakeArxivTool())

    result = arxiv_search.invoke({"query": "anything"})

    assert result.startswith("Error searching arXiv")
    assert "no network" in result


def test_arxiv_search_reports_missing_dependency_clearly(monkeypatch):
    import app.tools as tools_module

    monkeypatch.setattr(tools_module, "_arxiv_tool", None)
    monkeypatch.setattr(tools_module, "_arxiv_import_error", ImportError("no module"))

    result = arxiv_search.invoke({"query": "anything"})

    assert result.startswith("Error: arxiv_search is unavailable")


def test_stackexchange_search_returns_summary_on_success(monkeypatch):
    import app.tools as tools_module

    class _FakeStackExchangeTool:
        def invoke(self, args: dict) -> str:
            return f"SO summary about {args['query']}"

    monkeypatch.setattr(tools_module, "_stackexchange_tool", _FakeStackExchangeTool())

    result = stackexchange_search.invoke({"query": "reverse a list in python"})

    assert result == "SO summary about reverse a list in python"


def test_stackexchange_search_handles_network_errors_gracefully(monkeypatch):
    import app.tools as tools_module

    class _FakeStackExchangeTool:
        def invoke(self, args: dict) -> str:
            raise ConnectionError("no network")

    monkeypatch.setattr(tools_module, "_stackexchange_tool", _FakeStackExchangeTool())

    result = stackexchange_search.invoke({"query": "anything"})

    assert result.startswith("Error searching Stack Exchange")
    assert "no network" in result


def test_stackexchange_search_reports_missing_dependency_clearly(monkeypatch):
    import app.tools as tools_module

    monkeypatch.setattr(tools_module, "_stackexchange_tool", None)
    monkeypatch.setattr(
        tools_module, "_stackexchange_import_error", ImportError("no module")
    )

    result = stackexchange_search.invoke({"query": "anything"})

    assert result.startswith("Error: stackexchange_search is unavailable")


def test_fetch_webpage_rejects_non_http_url():
    result = fetch_webpage.invoke({"url": "not-a-url"})

    assert result.startswith("Error:")
    assert "not a valid http" in result


def test_fetch_webpage_strips_html_and_returns_text(monkeypatch):
    import app.tools as tools_module

    class _FakeResponse:
        content = (
            b"<html><head><style>body{}</style></head>"
            b"<body><p>Hello world</p></body></html>"
        )

        def raise_for_status(self) -> None:
            return None

    class _FakeRequestsModule:
        @staticmethod
        def get(url, timeout=None, headers=None):
            return _FakeResponse()

    monkeypatch.setitem(__import__("sys").modules, "requests", _FakeRequestsModule())

    result = fetch_webpage.invoke({"url": "https://example.com"})

    assert result == "Hello world"



def test_fetch_webpage_handles_network_errors_gracefully(monkeypatch):
    class _FakeRequestsModule:
        @staticmethod
        def get(url, timeout=None, headers=None):
            raise ConnectionError("no network")

    monkeypatch.setitem(__import__("sys").modules, "requests", _FakeRequestsModule())

    result = fetch_webpage.invoke({"url": "https://example.com"})

    assert result.startswith("Error fetching")
    assert "no network" in result


def test_crawl_website_rejects_non_http_url():
    result = crawl_website.invoke({"url": "not-a-url", "max_pages": 3})

    assert result.startswith("Error:")
    assert "not a valid http" in result


def test_crawl_website_follows_same_domain_links_and_respects_max_pages(monkeypatch):
    pages = {
        "https://example.com/": (
            b"<html><body><p>Home page</p>"
            b'<a href="/about">About</a>'
            b'<a href="https://other.com/x">External</a>'
            b"</body></html>"
        ),
        "https://example.com/about": (
            b"<html><body><p>About page</p>"
            b'<a href="/contact">Contact</a>'
            b"</body></html>"
        ),
        "https://example.com/contact": b"<html><body><p>Contact page</p></body></html>",
    }

    class _FakeResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

    class _FakeRequestsModule:
        @staticmethod
        def get(url, timeout=None, headers=None):
            return _FakeResponse(pages[url])

    monkeypatch.setitem(__import__("sys").modules, "requests", _FakeRequestsModule())

    result = crawl_website.invoke({"url": "https://example.com/", "max_pages": 2})

    assert "=== https://example.com/ ===" in result
    assert "Home page" in result
    assert "=== https://example.com/about ===" in result
    assert "About page" in result
    # max_pages=2 must stop before the 3rd discovered page, and must never
    # cross onto the external, different-domain link.
    assert "Contact page" not in result
    assert "other.com" not in result


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


def test_grep_search_finds_literal_match(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.py").write_text("def bar():\n    return 2\n")

    result = grep_search.invoke({"query": "def foo", "dir_path": str(tmp_path)})

    assert "a.py:1:" in result
    assert "b.py" not in result


def test_grep_search_regex_mode(tmp_path):
    (tmp_path / "a.py").write_text("value = 123\nother = 456\n")

    result = grep_search.invoke(
        {"query": r"\d{3}", "dir_path": str(tmp_path), "is_regex": True}
    )

    assert "a.py:1:" in result
    assert "a.py:2:" in result


def test_grep_search_respects_include_pattern(tmp_path):
    (tmp_path / "a.py").write_text("needle\n")
    (tmp_path / "b.txt").write_text("needle\n")

    result = grep_search.invoke(
        {"query": "needle", "dir_path": str(tmp_path), "include_pattern": "*.py"}
    )

    assert "a.py" in result
    assert "b.txt" not in result


def test_grep_search_no_matches(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")

    result = grep_search.invoke(
        {"query": "zzz_definitely_not_present_zzz", "dir_path": str(tmp_path)}
    )

    assert result.startswith("No matches found")


def test_grep_search_rejects_missing_directory():
    result = grep_search.invoke({"query": "x", "dir_path": "/no/such/dir"})
    assert result.startswith("Error:")


def test_grep_search_rejects_invalid_regex():
    result = grep_search.invoke({"query": "(", "is_regex": True})
    assert result.startswith("Error: invalid regex")


def test_get_errors_reports_clean_file(tmp_path):
    clean = tmp_path / "clean.py"
    clean.write_text("def add(a, b):\n    return a + b\n")

    result = get_errors.invoke({"file_paths": [str(clean)]})

    assert f"'{clean}': no errors found." in result


def test_get_errors_reports_real_problem(tmp_path):
    broken = tmp_path / "broken.py"
    # Unused import -- a real ruff (F401) finding.
    broken.write_text("import os\n\n\ndef f():\n    return 1\n")

    result = get_errors.invoke({"file_paths": [str(broken)]})

    assert "problem(s)" in result
    assert "F401" in result


def test_get_errors_reports_missing_file():
    result = get_errors.invoke({"file_paths": ["/no/such/file.py"]})
    assert "not found" in result


def test_get_errors_rejects_empty_list():
    result = get_errors.invoke({"file_paths": []})
    assert result.startswith("Error:")


def test_ask_user_echoes_question():
    result = ask_user.invoke({"question": "Which file do you mean?"})
    assert "Which file do you mean?" in result


