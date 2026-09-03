from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx
import psycopg
from openai import OpenAIError

from research_radar.agents import AgentService, paper_embedding_text
from research_radar.config import Settings
from research_radar.database import Repository
from research_radar.models import Topic, Usage
from research_radar.openalex import OpenAlexClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SyncResult:
    topic_slug: str
    status: str
    fetched: int = 0
    new: int = 0
    indexed: int = 0
    summarized: int = 0
    estimated_cost_usd: float = 0
    error: str | None = None


class SyncPipeline:
    def __init__(
        self, settings: Settings, repository: Repository, topics: tuple[Topic, ...]
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._topics = topics
        self._openalex = OpenAlexClient(settings)

    def run(self, *, topic_slug: str | None = None, force: bool = False) -> list[SyncResult]:
        selected_topics = self._select_topics(topic_slug)
        results: list[SyncResult] = []

        with self._repository.sync_lock() as acquired:
            if not acquired:
                return [
                    SyncResult(
                        topic_slug="all", status="skipped", error="sync already running"
                    )
                ]

            for topic in selected_topics:
                last_success = self._repository.last_success(topic.slug)
                if not force and not topic_is_due(last_success, topic.cadence_hours):
                    results.append(SyncResult(topic_slug=topic.slug, status="skipped"))
                    continue
                results.append(self._run_topic(topic, last_success))

        failures = [result for result in results if result.status == "failed"]
        if failures:
            names = ", ".join(result.topic_slug for result in failures)
            raise RuntimeError(f"Sync failed for: {names}. See the runs page for details.")
        return results

    def _select_topics(self, topic_slug: str | None) -> tuple[Topic, ...]:
        if topic_slug is None:
            return self._topics
        selected = tuple(topic for topic in self._topics if topic.slug == topic_slug)
        if not selected:
            available = ", ".join(topic.slug for topic in self._topics)
            raise ValueError(f"Unknown topic '{topic_slug}'. Available topics: {available}")
        return selected

    def _run_topic(self, topic: Topic, last_success: datetime | None) -> SyncResult:
        run_id = self._repository.start_run(topic.slug)
        usage = Usage()
        fetched_count = new_count = indexed_count = summarized_count = 0

        try:
            agent = AgentService(self._settings)
            self._repository.prepare_embeddings(topic.slug, agent.embedding_model_id)
            from_date = (
                last_success.date() - timedelta(days=2)
                if last_success
                else datetime.now(UTC).date() - timedelta(days=topic.lookback_days)
            )
            discovered = self._openalex.discover(topic, from_date)
            fetched_count = len(discovered)
            new_count = self._repository.upsert_papers(topic.slug, discovered)

            pending_curations = self._repository.pending_papers(
                topic.slug, ("discovered",), topic.max_candidates
            )
            if pending_curations:
                decisions, curator_usage = agent.curate(topic, pending_curations)
                usage += curator_usage
                selected_ids = {
                    decision.openalex_id
                    for decision in sorted(
                        decisions, key=lambda item: item.relevance_score, reverse=True
                    )[: topic.summarize_top]
                    if decision.relevance_score >= topic.min_relevance_score
                }
                self._repository.save_curations(topic.slug, decisions, selected_ids)

            pending_index = self._repository.pending_embeddings(
                topic.slug, topic.max_candidates
            )
            if pending_index:
                vectors, embedding_usage = agent.embed(
                    [paper_embedding_text(paper) for paper in pending_index]
                )
                usage += embedding_usage
                self._repository.save_embeddings(
                    topic.slug,
                    pending_index,
                    vectors,
                    agent.embedding_model_id,
                )
                indexed_count = len(pending_index)

            for paper in self._repository.pending_summaries(
                topic.slug, topic.summarize_top
            ):
                summary, analyst_usage = agent.summarize(topic, paper)
                usage += analyst_usage
                self._repository.save_summary(topic.slug, paper.openalex_id, summary)
                summarized_count += 1

            finished_at = datetime.now(UTC)
            estimated_cost = self._settings.estimate_cost(usage)
            self._repository.mark_source_success(topic.slug, finished_at)
            self._repository.finish_run(
                run_id,
                status="succeeded",
                fetched_count=fetched_count,
                new_count=new_count,
                indexed_count=indexed_count,
                summarized_count=summarized_count,
                usage=usage,
                estimated_cost_usd=estimated_cost,
            )
            return SyncResult(
                topic_slug=topic.slug,
                status="succeeded",
                fetched=fetched_count,
                new=new_count,
                indexed=indexed_count,
                summarized=summarized_count,
                estimated_cost_usd=estimated_cost,
            )
        except (OpenAIError, psycopg.Error, httpx.HTTPError, RuntimeError, ValueError) as exc:
            logger.exception("Sync failed for topic %s", topic.slug)
            self._repository.finish_run(
                run_id,
                status="failed",
                fetched_count=fetched_count,
                new_count=new_count,
                indexed_count=indexed_count,
                summarized_count=summarized_count,
                usage=usage,
                estimated_cost_usd=self._settings.estimate_cost(usage),
                error=str(exc)[:2000],
            )
            return SyncResult(topic_slug=topic.slug, status="failed", error=str(exc))


def topic_is_due(
    last_success: datetime | None,
    cadence_hours: int,
    now: datetime | None = None,
) -> bool:
    if last_success is None:
        return True
    current_time = now or datetime.now(UTC)
    return current_time - last_success >= timedelta(hours=cadence_hours)
