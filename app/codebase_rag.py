"""Retrieval over the project's OWN source code, built on the same
FAISS + Ollama-embeddings machinery as ``app.rag``'s company-facts KB.

Unlike ``app.rag.rag_search`` (a tiny hand-written set of company policy
docs), this module indexes the REAL files in this repository (``app/``,
``tests/``, and top-level docs/config like ``README.md``), chunked into
fixed-size, overlapping line windows. This lets the agent answer questions
like "do we use prompt templates in this project?" by semantically
searching its own source, rather than guessing, narrating a shell command
it never runs, or hallucinating a plausible-sounding non-answer.

The index is built lazily (on first search) and cached as a module-level
singleton, exactly like ``app.rag.KnowledgeBaseIndex`` -- so importing this
module (transitively via ``app.tools``) never triggers a filesystem walk or
embedding calls unless the tool is actually used.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from app.config import OLLAMA_BASE_URL
from app.rag import KnowledgeBaseIndex, RagUnavailableError

# Only index source-ish, human-readable files; skip caches/venvs/binaries.
INCLUDED_EXTENSIONS = {".py", ".md", ".txt", ".toml", ".ini", ".cfg", ".yml", ".yaml"}
EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".venv",
    "venv",
    ".git",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
}

CHUNK_LINES = 60  # lines per chunk -- small enough for a focused embedding
CHUNK_OVERLAP_LINES = 10  # overlap so a fact spanning a boundary isn't lost

# The project root is the parent of the "app" package this file lives in,
# resolved from __file__ so this works regardless of the current working
# directory the process happens to be started from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _iter_source_files(root: Path = PROJECT_ROOT) -> list[Path]:
    """List every indexable file under ``root``, skipping excluded dirs."""
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in INCLUDED_EXTENSIONS:
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        files.append(path)
    return files


def _chunk_file(path: Path, root: Path = PROJECT_ROOT) -> list[tuple[str, str]]:
    """Split one file into overlapping line-range chunks.

    Returns ``(doc_id, text)`` pairs where ``doc_id`` encodes the relative
    file path and the 1-based line range, e.g. ``"app/react_agent.py:1-60"``
    -- so the agent can tell the user exactly where a fact came from instead
    of just paraphrasing without a source.
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    if not lines:
        return []

    rel_path = path.relative_to(root).as_posix()
    chunks: list[tuple[str, str]] = []
    step = max(CHUNK_LINES - CHUNK_OVERLAP_LINES, 1)
    start = 0
    while start < len(lines):
        end = min(start + CHUNK_LINES, len(lines))
        text = "\n".join(lines[start:end])
        if text.strip():
            chunks.append((f"{rel_path}:{start + 1}-{end}", text))
        if end == len(lines):
            break
        start += step
    return chunks


def build_codebase_documents(root: Path = PROJECT_ROOT) -> list[tuple[str, str]]:
    """Build the full list of ``(doc_id, text)`` chunks for the whole project."""
    documents: list[tuple[str, str]] = []
    for file_path in _iter_source_files(root):
        documents.extend(_chunk_file(file_path, root))
    return documents


class _LazyCodebaseIndex:
    """Defers both the filesystem walk (``build_codebase_documents``) and
    the FAISS index build (one Ollama embedding call per chunk) until the
    first real search -- so merely importing this module stays instant and
    network-free, matching the laziness pattern in ``app.rag``.
    """

    def __init__(self, root: Path = PROJECT_ROOT, base_url: str = OLLAMA_BASE_URL) -> None:
        self._root = root
        self._base_url = base_url
        self._index: KnowledgeBaseIndex | None = None

    def search(self, query: str, k: int = 3) -> list[tuple[str, str, float]]:
        if self._index is None:
            documents = build_codebase_documents(self._root)
            self._index = KnowledgeBaseIndex(documents=documents, base_url=self._base_url)
        return self._index.search(query, k=k)


# Module-level singleton so the (potentially large) codebase index is only
# built once per process, not once per tool call.
_default_codebase_index = _LazyCodebaseIndex()


@tool
def codebase_search(query: str) -> str:
    """Semantically search THIS PROJECT'S OWN source code and docs -- NOT
    company policy facts (use 'rag_search' for those).

    Use this whenever asked anything about how this codebase itself is
    built, organized, or whether it implements a particular pattern -- e.g.
    "do we use prompt templates?", "how is conversation history handled?",
    "where is the ReAct loop implemented?", "what does run_shell_command
    do?". Returns the most relevant file(s) and line range(s) with the
    actual matching text, so you can quote or explain it directly instead
    of guessing, describing a shell command instead of running it, or
    hallucinating.

    Requires Ollama running locally with the 'nomic-embed-text' model pulled.
    """
    try:
        results = _default_codebase_index.search(query, k=3)
    except RagUnavailableError as exc:
        return f"Error: {exc}"

    if not results:
        return "No relevant source files found."

    return "\n\n".join(f"[{doc_id}]\n{text}" for doc_id, text, _distance in results)
