from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from research_radar.models import CuratorDecision, DiscoveredPaper, PaperSummary, Usage

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    openalex_id TEXT NOT NULL UNIQUE,
    doi TEXT,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors JSONB NOT NULL DEFAULT '[]'::jsonb,
    venue TEXT,
    published_at DATE NOT NULL,
    source_url TEXT NOT NULL,
    pdf_url TEXT,
    oa_license TEXT,
    cited_by_count INTEGER NOT NULL DEFAULT 0,
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(abstract, ''))
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS paper_topics (
    paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    topic_slug TEXT NOT NULL,
    relevance_score SMALLINT CHECK (relevance_score BETWEEN 0 AND 100),
    curation_reason TEXT,
    selected_for_summary BOOLEAN NOT NULL DEFAULT false,
    summary JSONB,
    tags TEXT[] NOT NULL DEFAULT '{}',
    embedding VECTOR,
    embedding_model TEXT,
    processing_status TEXT NOT NULL DEFAULT 'discovered'
        CHECK (processing_status IN ('discovered', 'curated', 'indexed', 'summarized')),
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (paper_id, topic_slug)
);

CREATE TABLE IF NOT EXISTS runs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    topic_slug TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    fetched_count INTEGER NOT NULL DEFAULT 0,
    new_count INTEGER NOT NULL DEFAULT 0,
    indexed_count INTEGER NOT NULL DEFAULT 0,
    summarized_count INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    embedding_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd NUMERIC(12, 6) NOT NULL DEFAULT 0,
    error TEXT
);

CREATE TABLE IF NOT EXISTS source_state (
    source TEXT NOT NULL,
    topic_slug TEXT NOT NULL,
    last_success_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (source, topic_slug)
);

CREATE INDEX IF NOT EXISTS papers_search_idx ON papers USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS paper_topics_feed_idx
    ON paper_topics (topic_slug, discovered_at DESC);
CREATE INDEX IF NOT EXISTS paper_topics_status_idx
    ON paper_topics (topic_slug, processing_status);

ALTER TABLE paper_topics ADD COLUMN IF NOT EXISTS embedding_model TEXT;
DO $$
BEGIN
    IF (
        SELECT atttypmod
        FROM pg_attribute
        WHERE attrelid = 'paper_topics'::regclass AND attname = 'embedding'
    ) <> -1 THEN
        ALTER TABLE paper_topics ALTER COLUMN embedding TYPE VECTOR;
    END IF;
END
$$;
"""


class Repository:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self) -> psycopg.Connection[dict[str, Any]]:
        return psycopg.connect(self._database_url, row_factory=dict_row)

    def init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(SCHEMA)

    def health(self) -> bool:
        with self._connect() as connection:
            return connection.execute("SELECT 1").fetchone() is not None

    @contextmanager
    def sync_lock(self) -> Iterator[bool]:
        connection = self._connect()
        acquired = False
        try:
            row = connection.execute(
                "SELECT pg_try_advisory_lock(824_701) AS acquired"
            ).fetchone()
            acquired = bool(row and row["acquired"])
            yield acquired
        finally:
            if acquired:
                connection.execute("SELECT pg_advisory_unlock(824_701)")
            connection.close()

    def last_success(self, topic_slug: str) -> datetime | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT last_success_at
                FROM source_state
                WHERE source = 'openalex' AND topic_slug = %s
                """,
                (topic_slug,),
            ).fetchone()
        return row["last_success_at"] if row else None

    def mark_source_success(self, topic_slug: str, finished_at: datetime) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO source_state (source, topic_slug, last_success_at)
                VALUES ('openalex', %s, %s)
                ON CONFLICT (source, topic_slug) DO UPDATE
                SET last_success_at = EXCLUDED.last_success_at
                """,
                (topic_slug, finished_at),
            )

    def start_run(self, topic_slug: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                INSERT INTO runs (topic_slug, source, status)
                VALUES (%s, 'openalex', 'running')
                RETURNING id
                """,
                (topic_slug,),
            ).fetchone()
        assert row is not None
        return int(row["id"])

    def finish_run(
        self,
        run_id: int,
        *,
        status: str,
        fetched_count: int = 0,
        new_count: int = 0,
        indexed_count: int = 0,
        summarized_count: int = 0,
        usage: Usage | None = None,
        estimated_cost_usd: float = 0,
        error: str | None = None,
    ) -> None:
        run_usage = usage or Usage()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = %s,
                    finished_at = now(),
                    fetched_count = %s,
                    new_count = %s,
                    indexed_count = %s,
                    summarized_count = %s,
                    input_tokens = %s,
                    output_tokens = %s,
                    embedding_tokens = %s,
                    estimated_cost_usd = %s,
                    error = %s
                WHERE id = %s
                """,
                (
                    status,
                    fetched_count,
                    new_count,
                    indexed_count,
                    summarized_count,
                    run_usage.input_tokens,
                    run_usage.output_tokens,
                    run_usage.embedding_tokens,
                    estimated_cost_usd,
                    error,
                    run_id,
                ),
            )

    def upsert_papers(self, topic_slug: str, papers: list[DiscoveredPaper]) -> int:
        new_count = 0
        with self._connect() as connection:
            for paper in papers:
                row = connection.execute(
                    """
                    INSERT INTO papers (
                        openalex_id, doi, title, abstract, authors, venue, published_at,
                        source_url, pdf_url, oa_license, cited_by_count
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (openalex_id) DO UPDATE SET
                        doi = EXCLUDED.doi,
                        title = EXCLUDED.title,
                        abstract = EXCLUDED.abstract,
                        authors = EXCLUDED.authors,
                        venue = EXCLUDED.venue,
                        published_at = EXCLUDED.published_at,
                        source_url = EXCLUDED.source_url,
                        pdf_url = EXCLUDED.pdf_url,
                        oa_license = EXCLUDED.oa_license,
                        cited_by_count = EXCLUDED.cited_by_count,
                        updated_at = now()
                    RETURNING id
                    """,
                    (
                        paper.openalex_id,
                        paper.doi,
                        paper.title,
                        paper.abstract,
                        Jsonb(paper.authors),
                        paper.venue,
                        paper.published_at,
                        paper.source_url,
                        paper.pdf_url,
                        paper.oa_license,
                        paper.cited_by_count,
                    ),
                ).fetchone()
                assert row is not None
                linked = connection.execute(
                    """
                    INSERT INTO paper_topics (paper_id, topic_slug)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                    RETURNING paper_id
                    """,
                    (row["id"], topic_slug),
                ).fetchone()
                new_count += int(linked is not None)
        return new_count

    def pending_papers(
        self, topic_slug: str, statuses: tuple[str, ...], limit: int
    ) -> list[DiscoveredPaper]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.openalex_id, p.doi, p.title, p.abstract, p.authors, p.venue,
                       p.published_at, p.source_url, p.pdf_url, p.oa_license,
                       p.cited_by_count
                FROM papers p
                JOIN paper_topics pt ON pt.paper_id = p.id
                WHERE pt.topic_slug = %s AND pt.processing_status = ANY(%s)
                ORDER BY pt.discovered_at
                LIMIT %s
                """,
                (topic_slug, list(statuses), limit),
            ).fetchall()
        return [DiscoveredPaper.model_validate(row) for row in rows]

    def save_curations(
        self,
        topic_slug: str,
        decisions: list[CuratorDecision],
        selected_ids: set[str],
    ) -> None:
        with self._connect() as connection:
            for decision in decisions:
                connection.execute(
                    """
                    UPDATE paper_topics pt
                    SET relevance_score = %s,
                        curation_reason = %s,
                        selected_for_summary = %s,
                        processing_status = 'curated',
                        updated_at = now()
                    FROM papers p
                    WHERE pt.paper_id = p.id
                      AND pt.topic_slug = %s
                      AND p.openalex_id = %s
                      AND pt.processing_status = 'discovered'
                    """,
                    (
                        decision.relevance_score,
                        decision.reason,
                        decision.openalex_id in selected_ids,
                        topic_slug,
                        decision.openalex_id,
                    ),
                )

    def save_embeddings(
        self,
        topic_slug: str,
        papers: list[DiscoveredPaper],
        vectors: list[list[float]],
        embedding_model: str,
    ) -> None:
        if len(papers) != len(vectors):
            raise ValueError("Each paper must have exactly one embedding")
        dimensions = {len(vector) for vector in vectors}
        if not vectors or dimensions == {0} or len(dimensions) != 1:
            raise ValueError("Embeddings must be non-empty and have matching dimensions")
        with self._connect() as connection:
            for paper, vector in zip(papers, vectors, strict=True):
                connection.execute(
                    """
                    UPDATE paper_topics pt
                    SET embedding = %s::vector,
                        embedding_model = %s,
                        processing_status = CASE
                            WHEN pt.summary IS NULL THEN 'indexed'
                            ELSE 'summarized'
                        END,
                        updated_at = now()
                    FROM papers p
                    WHERE pt.paper_id = p.id
                      AND pt.topic_slug = %s
                      AND p.openalex_id = %s
                    """,
                    (
                        json.dumps(vector),
                        embedding_model,
                        topic_slug,
                        paper.openalex_id,
                    ),
                )

    def prepare_embeddings(self, topic_slug: str, embedding_model: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE paper_topics
                SET embedding = NULL,
                    embedding_model = NULL,
                    processing_status = CASE
                        WHEN summary IS NULL THEN 'curated'
                        ELSE 'summarized'
                    END,
                    updated_at = now()
                WHERE topic_slug = %s
                  AND embedding IS NOT NULL
                  AND embedding_model IS DISTINCT FROM %s
                """,
                (topic_slug, embedding_model),
            )

    def pending_embeddings(self, topic_slug: str, limit: int) -> list[DiscoveredPaper]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.openalex_id, p.doi, p.title, p.abstract, p.authors, p.venue,
                       p.published_at, p.source_url, p.pdf_url, p.oa_license,
                       p.cited_by_count
                FROM papers p
                JOIN paper_topics pt ON pt.paper_id = p.id
                WHERE pt.topic_slug = %s
                  AND pt.embedding IS NULL
                  AND pt.processing_status IN ('curated', 'summarized')
                ORDER BY pt.discovered_at
                LIMIT %s
                """,
                (topic_slug, limit),
            ).fetchall()
        return [DiscoveredPaper.model_validate(row) for row in rows]

    def pending_summaries(self, topic_slug: str, limit: int) -> list[DiscoveredPaper]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.openalex_id, p.doi, p.title, p.abstract, p.authors, p.venue,
                       p.published_at, p.source_url, p.pdf_url, p.oa_license,
                       p.cited_by_count
                FROM papers p
                JOIN paper_topics pt ON pt.paper_id = p.id
                WHERE pt.topic_slug = %s
                  AND pt.selected_for_summary
                  AND pt.summary IS NULL
                  AND pt.processing_status = 'indexed'
                ORDER BY pt.relevance_score DESC, pt.discovered_at
                LIMIT %s
                """,
                (topic_slug, limit),
            ).fetchall()
        return [DiscoveredPaper.model_validate(row) for row in rows]

    def save_summary(
        self, topic_slug: str, openalex_id: str, summary: PaperSummary
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE paper_topics pt
                SET summary = %s,
                    tags = %s,
                    processing_status = 'summarized',
                    updated_at = now()
                FROM papers p
                WHERE pt.paper_id = p.id
                  AND pt.topic_slug = %s
                  AND p.openalex_id = %s
                """,
                (Jsonb(summary.model_dump()), summary.tags, topic_slug, openalex_id),
            )

    def dashboard_metrics(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT count(DISTINCT p.id) AS papers,
                       count(*) FILTER (WHERE p.published_at >= current_date - 7)
                           AS published_this_week,
                       count(*) FILTER (WHERE pt.summary IS NOT NULL) AS summaries,
                       coalesce(round(avg(pt.relevance_score)), 0) AS average_relevance
                FROM papers p
                JOIN paper_topics pt ON pt.paper_id = p.id
                """
            ).fetchone()
        return row or {
            "papers": 0,
            "published_this_week": 0,
            "summaries": 0,
            "average_relevance": 0,
        }

    def list_papers(
        self, *, topic_slug: str | None = None, query: str = "", limit: int = 60
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return connection.execute(
                """
                WITH matches AS (
                    SELECT p.id, p.title, p.authors, p.venue, p.published_at,
                           p.source_url, p.pdf_url, p.cited_by_count,
                           pt.topic_slug, pt.relevance_score, pt.curation_reason,
                           pt.summary, pt.tags, pt.discovered_at,
                           row_number() OVER (
                               PARTITION BY p.id
                               ORDER BY pt.relevance_score DESC NULLS LAST
                           ) AS paper_rank
                    FROM papers p
                    JOIN paper_topics pt ON pt.paper_id = p.id
                    WHERE (%s::text IS NULL OR pt.topic_slug = %s)
                      AND (%s = '' OR p.search_vector @@ websearch_to_tsquery('english', %s))
                )
                SELECT * FROM matches
                WHERE paper_rank = 1
                ORDER BY (summary IS NOT NULL)::int DESC,
                         published_at DESC,
                         relevance_score DESC NULLS LAST
                LIMIT %s
                """,
                (topic_slug, topic_slug, query, query, limit),
            ).fetchall()

    def get_paper(self, paper_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            paper = connection.execute(
                """
                SELECT id, openalex_id, doi, title, abstract, authors, venue,
                       published_at, source_url, pdf_url, oa_license, cited_by_count
                FROM papers
                WHERE id = %s
                """,
                (paper_id,),
            ).fetchone()
            if paper is None:
                return None
            paper["topics"] = connection.execute(
                """
                SELECT topic_slug, relevance_score, curation_reason, summary, tags,
                       selected_for_summary, processing_status, discovered_at
                FROM paper_topics
                WHERE paper_id = %s
                ORDER BY relevance_score DESC NULLS LAST
                """,
                (paper_id,),
            ).fetchall()
        return paper

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT id, topic_slug, source, status, started_at, finished_at,
                       fetched_count, new_count, indexed_count, summarized_count,
                       input_tokens, output_tokens, embedding_tokens,
                       estimated_cost_usd, error
                FROM runs
                ORDER BY started_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()

    def search(
        self,
        query: str,
        vector: list[float],
        embedding_model: str,
        *,
        topic_slug: str | None = None,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return connection.execute(
                """
                WITH candidates AS (
                    SELECT p.id, p.title, p.abstract, p.authors, p.published_at,
                           p.source_url, pt.topic_slug, pt.summary,
                           1 - (pt.embedding <=> %s::vector) AS semantic_score,
                           ts_rank_cd(
                               p.search_vector,
                               websearch_to_tsquery('english', %s)
                           ) AS lexical_score
                    FROM papers p
                    JOIN paper_topics pt ON pt.paper_id = p.id
                    WHERE pt.embedding IS NOT NULL
                      AND pt.embedding_model = %s
                      AND (%s::text IS NULL OR pt.topic_slug = %s)
                ), scored AS (
                    SELECT *, semantic_score * 0.85
                              + least(lexical_score * 5, 1) * 0.15 AS score
                    FROM candidates
                ), ranked AS (
                    SELECT *, row_number() OVER (
                        PARTITION BY id ORDER BY score DESC
                    ) AS paper_rank
                    FROM scored
                )
                SELECT * FROM ranked
                WHERE paper_rank = 1
                ORDER BY score DESC
                LIMIT %s
                """,
                (
                    json.dumps(vector),
                    query,
                    embedding_model,
                    topic_slug,
                    topic_slug,
                    limit,
                ),
            ).fetchall()
