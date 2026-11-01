from unittest.mock import patch

from app.codebase_rag import (
    CHUNK_LINES,
    _chunk_file,
    _iter_source_files,
    build_codebase_documents,
    codebase_search,
)
from app.rag import RagUnavailableError


def test_iter_source_files_includes_python_and_excludes_cache_dirs(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "foo.py").write_text("print('hi')\n")
    (tmp_path / "app" / "__pycache__").mkdir()
    (tmp_path / "app" / "__pycache__" / "foo.cpython-313.pyc").write_bytes(b"\x00\x01")
    (tmp_path / "README.md").write_text("# Title\n")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    files = _iter_source_files(tmp_path)

    rel_paths = {p.relative_to(tmp_path).as_posix() for p in files}
    assert rel_paths == {"app/foo.py", "README.md"}


def test_chunk_file_splits_long_file_into_overlapping_chunks(tmp_path):
    total_lines = CHUNK_LINES + 20
    content = "\n".join(f"line {i}" for i in range(total_lines))
    file_path = tmp_path / "big.py"
    file_path.write_text(content)

    chunks = _chunk_file(file_path, root=tmp_path)

    assert len(chunks) >= 2
    first_doc_id, first_text = chunks[0]
    assert first_doc_id.startswith("big.py:1-")
    assert "line 0" in first_text
    # The last chunk should reach the end of the file.
    last_doc_id, last_text = chunks[-1]
    assert f"line {total_lines - 1}" in last_text
    assert last_doc_id.endswith(f"-{total_lines}")


def test_chunk_file_skips_empty_files(tmp_path):
    file_path = tmp_path / "empty.py"
    file_path.write_text("")

    assert _chunk_file(file_path, root=tmp_path) == []


def test_build_codebase_documents_covers_multiple_files(tmp_path):
    (tmp_path / "a.py").write_text("def a():\n    return 1\n")
    (tmp_path / "b.md").write_text("# docs\n")

    documents = build_codebase_documents(tmp_path)

    doc_ids = {doc_id for doc_id, _ in documents}
    assert any(doc_id.startswith("a.py:") for doc_id in doc_ids)
    assert any(doc_id.startswith("b.md:") for doc_id in doc_ids)


def test_codebase_search_tool_returns_formatted_matches():
    fake_results = [
        ("app/react_agent.py:1-60", "class ReActAgent:\n    ...", 0.1),
        ("app/prompts.py:1-60", "REACT_SYSTEM_TEMPLATE = ...", 0.2),
    ]
    with patch("app.codebase_rag._default_codebase_index.search", return_value=fake_results):
        result = codebase_search.invoke({"query": "how does the ReAct loop work"})

    assert "[app/react_agent.py:1-60]" in result
    assert "class ReActAgent:" in result
    assert "[app/prompts.py:1-60]" in result


def test_codebase_search_tool_handles_unavailable_embedding_backend():
    with patch(
        "app.codebase_rag._default_codebase_index.search",
        side_effect=RagUnavailableError("Ollama is down"),
    ):
        result = codebase_search.invoke({"query": "anything"})

    assert result == "Error: Ollama is down"


def test_codebase_search_tool_handles_no_results():
    with patch("app.codebase_rag._default_codebase_index.search", return_value=[]):
        result = codebase_search.invoke({"query": "anything"})

    assert result == "No relevant source files found."
