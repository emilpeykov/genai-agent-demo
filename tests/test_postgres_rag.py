from unittest.mock import MagicMock, patch

import pytest

import app.postgres_rag as postgres_rag
from app.postgres_rag import (
    PostgresRagUnavailableError,
    _get_connection,
    add_embedding_column,
    article_search,
    backfill_embeddings,
    search_articles,
    upsert_article,
)


def _make_fake_connection(fetch_return=None):
    """Build a MagicMock that behaves like a psycopg connection/cursor pair
    used via ``with _get_connection() as conn, conn.cursor() as cur:``.
    """
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False
    cursor.fetchall.return_value = fetch_return or []

    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    conn.cursor.return_value = cursor
    return conn, cursor


def test_get_connection_raises_when_dsn_not_set(monkeypatch):
    monkeypatch.setattr(postgres_rag, "POSTGRES_DSN", None)

    with pytest.raises(PostgresRagUnavailableError, match="POSTGRES_DSN"):
        _get_connection()


def test_get_connection_raises_when_psycopg_not_installed(monkeypatch):
    monkeypatch.setattr(postgres_rag, "POSTGRES_DSN", "postgresql://fake-dsn")

    # psycopg is intentionally not a hard dependency of this project -- this
    # exercises the real "not installed" path rather than mocking it away.
    with pytest.raises(PostgresRagUnavailableError, match="psycopg"):
        _get_connection()


def test_upsert_article_embeds_content_and_runs_upsert_sql():
    conn, cursor = _make_fake_connection()
    with (
        patch("app.postgres_rag._get_connection", return_value=conn),
        patch("app.postgres_rag._embed", return_value=[0.1, 0.2]) as mock_embed,
    ):
        upsert_article(1, "My Title", "My content")

    mock_embed.assert_called_once_with("My content", postgres_rag.OLLAMA_BASE_URL)
    cursor.execute.assert_called_once()
    sql, params = cursor.execute.call_args[0]
    assert "INSERT INTO articles" in sql
    assert "ON CONFLICT" in sql
    assert params == (1, "My Title", "My content", [0.1, 0.2])
    conn.commit.assert_called_once()


def test_search_articles_returns_parsed_rows_closest_first():
    conn, cursor = _make_fake_connection(fetch_return=[(1, "Title", "Content", 0.05)])
    with (
        patch("app.postgres_rag._get_connection", return_value=conn),
        patch("app.postgres_rag._embed", return_value=[0.1, 0.2]),
    ):
        results = search_articles("my query", k=3)

    assert results == [(1, "Title", "Content", 0.05)]
    sql, params = cursor.execute.call_args[0]
    assert "ORDER BY embedding <->" in sql
    assert params[-1] == 3


def test_article_search_tool_formats_matches():
    fake_results = [(1, "Launch Checklist", "Some article content here" * 20, 0.1)]
    with patch("app.postgres_rag.search_articles", return_value=fake_results):
        result = article_search.invoke({"query": "how do we launch"})

    assert "[article #1] Launch Checklist" in result


def test_article_search_tool_handles_unavailable_backend():
    with patch(
        "app.postgres_rag.search_articles",
        side_effect=PostgresRagUnavailableError("POSTGRES_DSN is not set"),
    ):
        result = article_search.invoke({"query": "anything"})

    assert result == "Error: POSTGRES_DSN is not set"


def test_article_search_tool_handles_no_results():
    with patch("app.postgres_rag.search_articles", return_value=[]):
        result = article_search.invoke({"query": "anything"})

    assert result == "No relevant articles found."


def test_add_embedding_column_enables_extension_and_adds_column():
    conn, cursor = _make_fake_connection()
    with patch("app.postgres_rag._get_connection", return_value=conn):
        add_embedding_column(table_name="my_existing_articles", embedding_column="my_embedding")

    executed_sql = [call_args[0][0] for call_args in cursor.execute.call_args_list]
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in sql for sql in executed_sql)
    assert any(
        "ALTER TABLE my_existing_articles" in sql and "ADD COLUMN IF NOT EXISTS my_embedding" in sql
        for sql in executed_sql
    )
    assert any("CREATE INDEX IF NOT EXISTS" in sql for sql in executed_sql)
    conn.commit.assert_called_once()


def test_backfill_embeddings_embeds_only_rows_missing_an_embedding():
    conn, cursor = _make_fake_connection()
    # First batch: one row needing embedding. Second call: no more rows left.
    cursor.fetchall.side_effect = [[(42, "Some article content")], []]

    with (
        patch("app.postgres_rag._get_connection", return_value=conn),
        patch("app.postgres_rag._embed", return_value=[0.1, 0.2]) as mock_embed,
    ):
        embedded_count = backfill_embeddings(
            table_name="my_existing_articles",
            id_column="article_id",
            content_column="body",
            embedding_column="my_embedding",
        )

    assert embedded_count == 1
    mock_embed.assert_called_once_with("Some article content", postgres_rag.OLLAMA_BASE_URL)

    select_sql = cursor.execute.call_args_list[0][0][0]
    assert "my_existing_articles" in select_sql
    assert "my_embedding IS NULL" in select_sql

    update_sql, update_params = cursor.execute.call_args_list[1][0]
    assert "UPDATE my_existing_articles SET my_embedding" in update_sql
    assert update_params == ([0.1, 0.2], 42)


def test_backfill_embeddings_returns_zero_when_nothing_to_embed():
    conn, cursor = _make_fake_connection(fetch_return=[])

    with patch("app.postgres_rag._get_connection", return_value=conn):
        embedded_count = backfill_embeddings()

    assert embedded_count == 0
