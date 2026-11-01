"""A dedicated RAG index built from ONLY the project's README.md file.

This is a small, focused demonstration of the core RAG pattern end-to-end,
scoped to a single document instead of the whole repo
(``app.codebase_rag``) or a hardcoded fact list (``app.rag``):

  1. Load README.md and split it into chunks along its own ``## section``
     headers (so each chunk is a coherent, self-contained unit -- "## Running
     locally" stays together, rather than being cut mid-sentence by a fixed
     line-count window).
  2. Embed each chunk with Ollama's ``nomic-embed-text`` model and store the
     vectors in a FAISS index (reusing ``app.rag.KnowledgeBaseIndex``, the
     same machinery used for the company-facts and whole-codebase indexes).
  3. At query time, embed the query the same way and retrieve the closest
     section(s) by vector distance.

Why this exists as its OWN index (rather than just relying on
``app.codebase_rag.codebase_search``, which already covers README.md as part
of indexing the whole repo): a dedicated, small index scoped to one document
is far less likely to have a query accidentally match an unrelated file
elsewhere in the repo -- exactly the failure mode observed when
"check the README" previously top-matched ``app/codebase_rag.py`` instead.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from app.config import OLLAMA_BASE_URL
from app.rag import KnowledgeBaseIndex, RagUnavailableError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README_PATH = PROJECT_ROOT / "README.md"


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown text into ``(section_title, section_text)`` chunks.

    A new chunk starts at every line beginning with ``#`` (any heading
    level). Content before the first heading (if any) is kept as a
    "(preamble)" chunk so nothing is silently dropped.
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = "(preamble)"
    current_lines: list[str] = []

    for line in lines:
        if line.lstrip().startswith("#"):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line.lstrip("#").strip() or current_title
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))

    return [
        (title, "\n".join(chunk_lines).strip())
        for title, chunk_lines in sections
        if "\n".join(chunk_lines).strip()
    ]


def build_readme_documents(path: Path = README_PATH) -> list[tuple[str, str]]:
    """Build ``(doc_id, text)`` chunks for the README, one per section."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    return _split_into_sections(text)


class _LazyReadmeIndex:
    """Defers both the file read and the FAISS index build (one embedding
    call per section) until the first real search, matching the laziness
    pattern used by ``app.rag`` and ``app.codebase_rag``.
    """

    def __init__(self, path: Path = README_PATH, base_url: str = OLLAMA_BASE_URL) -> None:
        self._path = path
        self._base_url = base_url
        self._index: KnowledgeBaseIndex | None = None

    def search(self, query: str, k: int = 2) -> list[tuple[str, str, float]]:
        if self._index is None:
            documents = build_readme_documents(self._path)
            self._index = KnowledgeBaseIndex(documents=documents, base_url=self._base_url)
        return self._index.search(query, k=k)


# Module-level singleton so the README index is only built once per process.
_default_readme_index = _LazyReadmeIndex()


@tool
def readme_search(query: str) -> str:
    """Semantically search ONLY this project's README.md file.

    Use this whenever the user asks what this project does, how to run it,
    its architecture, or anything else that the README would cover -- e.g.
    "what does this project do?", "how do I run it locally?", "what's the
    routing logic?". This is a small, dedicated RAG index built from just
    README.md's own sections, so it's much less likely to accidentally match
    an unrelated source file than 'codebase_search' (which searches the
    entire repo).

    Requires Ollama running locally with the 'nomic-embed-text' model pulled.
    """
    try:
        results = _default_readme_index.search(query, k=2)
    except RagUnavailableError as exc:
        return f"Error: {exc}"

    if not results:
        return "No relevant README section found."

    return "\n\n".join(f"[README.md \u00a7 {doc_id}]\n{text}" for doc_id, text, _distance in results)
