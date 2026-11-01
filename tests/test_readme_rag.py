from unittest.mock import patch

from app.rag import RagUnavailableError
from app.readme_rag import _split_into_sections, build_readme_documents, readme_search


def test_split_into_sections_splits_on_headings():
    text = (
        "# Title\n"
        "Intro text.\n"
        "## Section One\n"
        "Content one.\n"
        "## Section Two\n"
        "Content two.\n"
    )

    sections = _split_into_sections(text)

    titles = [title for title, _ in sections]
    assert "Title" in titles
    assert "Section One" in titles
    assert "Section Two" in titles

    section_two_text = dict(sections)["Section Two"]
    assert "Content two." in section_two_text


def test_split_into_sections_keeps_preamble_before_first_heading():
    text = "Some intro line with no heading yet.\n# First heading\nBody.\n"

    sections = _split_into_sections(text)

    assert sections[0][0] == "(preamble)"
    assert "Some intro line" in sections[0][1]


def test_split_into_sections_skips_empty_chunks():
    text = "# Title\n\n\n"

    sections = _split_into_sections(text)

    assert all(text.strip() for _, text in sections)


def test_build_readme_documents_reads_real_file(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# My Project\n\nThis project does X.\n## Running\nRun it with Y.\n")

    documents = build_readme_documents(readme)

    doc_ids = [doc_id for doc_id, _ in documents]
    assert "My Project" in doc_ids
    assert "Running" in doc_ids


def test_build_readme_documents_handles_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.md"
    assert build_readme_documents(missing) == []


def test_readme_search_tool_returns_formatted_section_matches():
    fake_results = [("Running locally", "cd genai-agent-demo\npip install -r requirements.txt", 0.1)]
    with patch("app.readme_rag._default_readme_index.search", return_value=fake_results):
        result = readme_search.invoke({"query": "how do I run this"})

    assert "[README.md \u00a7 Running locally]" in result
    assert "pip install -r requirements.txt" in result


def test_readme_search_tool_handles_unavailable_embedding_backend():
    with patch(
        "app.readme_rag._default_readme_index.search",
        side_effect=RagUnavailableError("Ollama is down"),
    ):
        result = readme_search.invoke({"query": "anything"})

    assert result == "Error: Ollama is down"


def test_readme_search_tool_handles_no_results():
    with patch("app.readme_rag._default_readme_index.search", return_value=[]):
        result = readme_search.invoke({"query": "anything"})

    assert result == "No relevant README section found."
