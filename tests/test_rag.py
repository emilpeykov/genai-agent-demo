from unittest.mock import patch

from app.rag import KnowledgeBaseIndex, RagUnavailableError, rag_search


def _fake_embed(text: str, base_url: str | None = None) -> list[float]:
    # Deterministic, hand-crafted 2D vectors so nearest-neighbor search has
    # an unambiguous, verifiable answer without needing a real Ollama server.
    if "cats" in text or "dogs" in text or "pets" in text:
        return [1.0, 0.0]
    if "stock" in text or "market" in text or "investing" in text:
        return [0.0, 1.0]
    return [0.5, 0.5]


def test_knowledge_base_index_finds_closest_document():
    docs = [("animals", "cats and dogs are common pets"), ("finance", "stock market trends")]

    with patch("app.rag._embed", side_effect=_fake_embed):
        index = KnowledgeBaseIndex(documents=docs)
        results = index.search("I am looking for new pets", k=1)

    assert len(results) == 1
    doc_id, text, distance = results[0]
    assert doc_id == "animals"
    assert "pets" in text
    assert distance == 0.0  # exact vector match in this crafted example


def test_knowledge_base_index_can_return_multiple_results():
    docs = [("animals", "cats and dogs"), ("finance", "stock market investing")]

    with patch("app.rag._embed", side_effect=_fake_embed):
        index = KnowledgeBaseIndex(documents=docs)
        results = index.search("investing", k=2)

    assert len(results) == 2
    assert {doc_id for doc_id, _, _ in results} == {"animals", "finance"}


def test_rag_search_tool_returns_matched_document_text():
    with patch("app.rag._default_index.search", return_value=[("sla", "4 hour response time", 0.1)]):
        result = rag_search.invoke({"query": "how fast do you respond"})

    assert result == "[sla] 4 hour response time"


def test_rag_search_tool_handles_unavailable_embedding_backend():
    with patch("app.rag._default_index.search", side_effect=RagUnavailableError("Ollama is down")):
        result = rag_search.invoke({"query": "anything"})

    assert result == "Error: Ollama is down"


def test_rag_search_tool_handles_no_results():
    with patch("app.rag._default_index.search", return_value=[]):
        result = rag_search.invoke({"query": "anything"})

    assert result == "No relevant internal documents found."
