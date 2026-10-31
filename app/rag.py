"""Minimal FAISS-backed retrieval (RAG) over a small in-memory document set.

Demonstrates the real mechanism behind LangChain/LlamaIndex-style retrieval
tools, as opposed to ``knowledge_base_lookup``'s hardcoded dict:

  1. Each document is turned into an embedding vector (via Ollama's
     'nomic-embed-text' model -- already pulled locally, no API key needed).
  2. All vectors are stored in a FAISS index (an in-memory similarity index,
     the same library used in most real-world LangChain/LlamaIndex RAG
     pipelines before upgrading to a hosted vector DB like Pinecone/Weaviate).
  3. A query is embedded the same way, then FAISS finds the closest document
     vectors by distance -- this generalizes to paraphrases and synonyms
     that an exact-string-match dict could never catch.

``faiss`` is imported lazily (inside ``_ensure_built``) so simply importing
this module -- e.g. transitively via ``app.tools`` -- never requires the
(fairly heavy) faiss/numpy stack unless the RAG tool is actually invoked.
"""
from __future__ import annotations

import httpx
from langchain_core.tools import tool

from app.config import OLLAMA_BASE_URL

EMBEDDING_MODEL = "nomic-embed-text"

# A slightly richer version of the facts in app.tools.knowledge_base_lookup,
# plus a couple of new ones -- enough to show real nearest-neighbor behavior
# (e.g. "how fast do you respond to tickets" should still find the SLA doc).
DOCUMENTS: list[tuple[str, str]] = [
    (
        "refund_policy",
        "Refunds are processed within 5-7 business days of approval. "
        "Customers must request a refund within 30 days of the original purchase.",
    ),
    (
        "sla",
        "Standard support SLA is a 4 business hour first response time during "
        "business hours, Monday to Friday. Enterprise customers get 1 hour.",
    ),
    (
        "pricing",
        "Pricing tiers are Starter, Growth, and Enterprise, billed monthly. "
        "Enterprise includes a dedicated account manager and custom contracts.",
    ),
    (
        "onboarding",
        "New customers are onboarded via a guided setup call within the first "
        "week of signing up, followed by a 30-day check-in with their account manager.",
    ),
    (
        "security",
        "All customer data is encrypted at rest (AES-256) and in transit "
        "(TLS 1.2+). The platform is SOC 2 Type II compliant.",
    ),
]


class RagUnavailableError(Exception):
    """Raised when the embedding backend (Ollama) can't be reached."""


def _embed(text: str, base_url: str = OLLAMA_BASE_URL) -> list[float]:
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        raise RagUnavailableError(
            f"Could not reach Ollama embeddings API at {base_url} "
            f"(is 'ollama serve' running and has '{EMBEDDING_MODEL}' been pulled?): {exc}"
        ) from exc

    embedding = data.get("embedding")
    if not embedding:
        raise RagUnavailableError("Ollama returned no embedding for the given text")
    return embedding


class KnowledgeBaseIndex:
    """Lazily-built FAISS index over a set of (doc_id, text) documents.

    The index is only built (i.e. Ollama is only called to embed every
    document) the first time ``search`` is actually used -- not at import
    time -- so importing this module stays fast and network-free.
    """

    def __init__(
        self,
        documents: list[tuple[str, str]] = DOCUMENTS,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self._documents = documents
        self._base_url = base_url
        self._index = None  # type: ignore[var-annotated]

    def _ensure_built(self) -> None:
        if self._index is not None:
            return
        import faiss  # lazy import: only needed once the index is actually built
        import numpy as np

        vectors = [_embed(text, self._base_url) for _, text in self._documents]
        matrix = np.array(vectors, dtype="float32")
        index = faiss.IndexFlatL2(matrix.shape[1])
        index.add(matrix)
        self._index = index

    def search(self, query: str, k: int = 1) -> list[tuple[str, str, float]]:
        """Return up to ``k`` ``(doc_id, text, distance)`` tuples, closest first."""
        import numpy as np

        self._ensure_built()
        query_vector = np.array([_embed(query, self._base_url)], dtype="float32")
        distances, indices = self._index.search(query_vector, min(k, len(self._documents)))

        results: list[tuple[str, str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            doc_id, text = self._documents[idx]
            results.append((doc_id, text, float(dist)))
        return results


# Module-level singleton so the FAISS index (and its embeddings) are only
# computed once per process, not once per tool call.
_default_index = KnowledgeBaseIndex()


@tool
def rag_search(query: str) -> str:
    """Semantically search an internal knowledge base using FAISS + embeddings.

    Unlike 'knowledge_base_lookup' (which only matches exact known topic
    strings like 'refund policy'), this finds the most semantically similar
    internal document to ANY phrasing of a question -- e.g. "how quickly do
    you respond to support tickets" will still find the SLA document. Use
    this whenever asked about internal company facts/policies and the exact
    topic keyword isn't obviously one of a small fixed set.

    Requires Ollama running locally with the 'nomic-embed-text' model pulled.
    """
    try:
        results = _default_index.search(query, k=1)
    except RagUnavailableError as exc:
        return f"Error: {exc}"

    if not results:
        return "No relevant internal documents found."

    doc_id, text, _distance = results[0]
    return f"[{doc_id}] {text}"
