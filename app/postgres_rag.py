"""Postgres + pgvector-backed RAG -- the production-scale version of the
same pattern used by ``app.rag`` (hardcoded facts) and ``app.codebase_rag``
(whole-repo files), for when there are far too many documents to hold in
memory or re-embed every time the process starts.

Key architectural differences from the in-memory FAISS indexes elsewhere in
this project:

  - **Storage**: embeddings live in a normal Postgres table (one row per
    article), via the `pgvector` extension's ``vector`` column type --
    instead of an in-process ``faiss.IndexFlatL2`` that must be rebuilt
    (re-embedding everything!) every time the process restarts.
  - **Write path**: embedding happens ONCE, when an article is
    inserted/updated (``upsert_article``) -- not lazily on first search of
    the whole dataset. This means adding the 1,000,001st article doesn't
    require re-embedding the first million.
  - **Search path**: nearest-neighbor search runs as a single SQL query
    using pgvector's ``<->`` (distance) operator, with an ANN index
    (ivfflat/HNSW) doing the heavy lifting server-side -- it scales to
    millions of rows without ever loading all vectors into Python memory,
    unlike ``KnowledgeBaseIndex`` (used by ``app.rag``/``app.codebase_rag``/
    ``app.readme_rag``), which holds every vector in an in-memory FAISS index.

Requires (NOT installed by default -- this is an illustrative example, not a
hard dependency of the rest of the app):

  - A reachable Postgres instance with ``CREATE EXTENSION vector;`` applied.
  - ``POSTGRES_DSN`` set, e.g. ``postgresql://user:pass@localhost:5432/mydb``.
  - The ``psycopg[binary]`` package installed (``pip install "psycopg[binary]"``).

``psycopg`` is imported lazily (inside ``_get_connection``) so merely
importing this module -- e.g. transitively via ``app.tools`` -- never
requires it or a live database connection.
"""
from __future__ import annotations

from langchain_core.tools import tool

from app.config import OLLAMA_BASE_URL, POSTGRES_DSN
from app.rag import _embed

# Run this once (e.g. via a migration/setup script), not on every request.
ARTICLES_TABLE_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS articles (
    id BIGINT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768) NOT NULL
);

-- Approximate-nearest-neighbor index: makes search fast even over millions
-- of rows, at the cost of occasionally skipping a true nearest neighbor --
-- a standard, deliberate tradeoff at this scale (the same idea as FAISS's
-- approximate index types, e.g. IndexIVFFlat, vs. the exact IndexFlatL2
-- used elsewhere in this project for tiny in-memory document sets).
CREATE INDEX IF NOT EXISTS articles_embedding_idx
    ON articles USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
"""


class PostgresRagUnavailableError(Exception):
    """Raised when Postgres/pgvector isn't configured, installed, or reachable."""


def _get_connection():
    if not POSTGRES_DSN:
        raise PostgresRagUnavailableError(
            "POSTGRES_DSN is not set -- configure it to point at a Postgres "
            "instance with the pgvector extension installed, e.g. "
            "'postgresql://user:pass@localhost:5432/mydb'."
        )
    try:
        import psycopg  # lazy, optional import -- see module docstring
    except ImportError as exc:
        raise PostgresRagUnavailableError(
            "The 'psycopg[binary]' package is not installed -- run "
            "pip install \"psycopg[binary]\" to use Postgres-backed RAG."
        ) from exc

    try:
        return psycopg.connect(POSTGRES_DSN)
    except Exception as exc:  # pragma: no cover - requires a real/broken DB
        raise PostgresRagUnavailableError(f"Could not connect to Postgres: {exc}") from exc


def ensure_schema() -> None:
    """Create the pgvector extension, table, and ANN index if missing.

    Use this ONLY if you're starting from scratch (no existing articles
    table yet). If you already have a Postgres table full of articles, use
    ``add_embedding_column`` + ``backfill_embeddings`` instead -- they adapt
    to YOUR existing table/column names rather than creating a new one.

    Idempotent (everything is ``IF NOT EXISTS``) -- safe to call repeatedly,
    but intended to run once during setup/deployment, not per-request.
    """
    with _get_connection() as conn, conn.cursor() as cur:
        cur.execute(ARTICLES_TABLE_SCHEMA)
        conn.commit()


# ---------------------------------------------------------------------------
# Migrating an EXISTING table into RAG (no new database, no data movement).
#
# If you already have a Postgres database with an articles table, you do NOT
# need to create a new database or copy your data anywhere. You only need
# to: (1) turn on the pgvector extension, (2) add one new nullable column to
# your existing table to hold each row's embedding, and (3) run a one-time
# backfill to compute and fill in that column for your existing rows. Your
# original columns, rows, and ids are completely untouched.
#
# NOTE: ``table_name``/``id_column``/``content_column``/``embedding_column``
# are interpolated directly into SQL below (psycopg's ``%s`` placeholders
# can only parameterize VALUES, not identifiers like table/column names).
# These must come from your own trusted config -- never from end-user input.
# ---------------------------------------------------------------------------


def add_embedding_column(
    table_name: str = "articles",
    embedding_column: str = "embedding",
    embedding_dimensions: int = 768,
) -> None:
    """Enable pgvector and add an embedding column to an EXISTING table.

    Safe to call repeatedly. Does not touch any of your existing columns or
    rows -- it only adds one new, initially-NULL column (plus an ANN index)
    alongside what's already there.
    """
    with _get_connection() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"ALTER TABLE {table_name} "  # noqa: S608 - identifiers, not user input; see module note
            f"ADD COLUMN IF NOT EXISTS {embedding_column} vector({embedding_dimensions});"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {table_name}_{embedding_column}_idx "
            f"ON {table_name} USING ivfflat ({embedding_column} vector_l2_ops) WITH (lists = 100);"
        )
        conn.commit()


def backfill_embeddings(
    table_name: str = "articles",
    id_column: str = "id",
    content_column: str = "content",
    embedding_column: str = "embedding",
    batch_size: int = 50,
    base_url: str = OLLAMA_BASE_URL,
) -> int:
    """Embed and fill in ``embedding_column`` for existing rows that don't
    have one yet -- the one-time (or periodic) step that actually brings
    your pre-existing articles into RAG, without copying them to a new
    table or database.

    Only rows where ``embedding_column IS NULL`` are processed, so this is
    safe to re-run: interrupt it anytime and rerun later to pick up where it
    left off, or schedule it (e.g. as a cron job / nightly task) to embed
    newly-inserted articles going forward.

    Returns the number of rows embedded.
    """
    total = 0
    with _get_connection() as conn:
        while True:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {id_column}, {content_column} FROM {table_name} "
                    f"WHERE {embedding_column} IS NULL LIMIT %s",
                    (batch_size,),
                )
                rows = cur.fetchall()
            if not rows:
                break
            for row_id, content in rows:
                embedding = _embed(content, base_url)
                with conn.cursor() as cur:
                    cur.execute(
                        f"UPDATE {table_name} SET {embedding_column} = %s WHERE {id_column} = %s",
                        (embedding, row_id),
                    )
                conn.commit()
                total += 1
    return total


def upsert_article(article_id: int, title: str, content: str, base_url: str = OLLAMA_BASE_URL) -> None:
    """Embed one article's content and insert/update it in Postgres.

    Call this once per article, when it is created or edited -- this is the
    key difference from the in-memory indexes elsewhere in this project,
    which re-embed their ENTIRE document set the first time anyone searches.
    Here, embedding cost is paid once per write, no matter how many articles
    already exist or how many searches happen afterward.
    """
    embedding = _embed(content, base_url)
    with _get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO articles (id, title, content, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE
                SET title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding
            """,
            (article_id, title, content, embedding),
        )
        conn.commit()


def search_articles(
    query: str, k: int = 3, base_url: str = OLLAMA_BASE_URL
) -> list[tuple[int, str, str, float]]:
    """Return up to ``k`` ``(id, title, content, distance)`` rows, closest first.

    pgvector's ``<->`` operator is the SQL equivalent of FAISS's
    ``IndexFlatL2.search()`` -- except the nearest-neighbor math runs
    server-side in Postgres (accelerated by the ivfflat index), so it never
    requires loading the whole table's vectors into this process's memory.
    """
    query_embedding = _embed(query, base_url)
    with _get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, content, embedding <-> %s::vector AS distance
            FROM articles
            ORDER BY embedding <-> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, k),
        )
        rows = cur.fetchall()
    return [(row[0], row[1], row[2], float(row[3])) for row in rows]


@tool
def article_search(query: str) -> str:
    """Semantically search a (potentially huge) Postgres-backed articles table.

    Unlike 'rag_search'/'codebase_search'/'readme_search' (all small,
    in-memory FAISS indexes that get fully rebuilt on first use), this
    queries a persistent Postgres table via the pgvector extension --
    appropriate when there are too many articles to hold in memory, or new
    articles are added continuously. Requires POSTGRES_DSN to be configured
    and the psycopg package installed; otherwise returns a clear error
    explaining what's missing.
    """
    try:
        results = search_articles(query, k=3)
    except PostgresRagUnavailableError as exc:
        return f"Error: {exc}"

    if not results:
        return "No relevant articles found."

    return "\n\n".join(
        f"[article #{article_id}] {title}\n{content[:300]}"
        for article_id, title, content, _distance in results
    )
