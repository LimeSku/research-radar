"""Real PostgreSQL/pgvector smoke test; external API responses are synthetic."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import httpx
import psycopg
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.conninfo import make_conninfo

from research_radar import web
from research_radar.config import Settings
from research_radar.database import Repository
from research_radar.models import CuratorBatch, PaperSummary, ResearchAnswer, Topic
from research_radar.pipeline import SyncPipeline


class PipelineSmokeTest(unittest.TestCase):
    def test_pipeline_recovery_and_ask(self) -> None:
        database_url = os.environ["TEST_DATABASE_URL"]
        database_name = f"radar_smoke_{uuid4().hex}"
        with psycopg.connect(database_url, autocommit=True) as admin:
            admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
            try:
                self.check_pipeline(make_conninfo(database_url, dbname=database_name))
            finally:
                admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database_name)))

    def check_pipeline(self, database_url: str) -> None:
        settings = Settings(
            _env_file=None,
            database_url=database_url,
            ai_provider="ollama",
            ai_curation_batch_size=2,
            enable_public_ask=True,
        )
        topic = Topic(
            slug="smoke-test",
            name="Agent memory",
            description="Agent memory and planning.",
            query="agent memory",
            cadence_hours=6,
            lookback_days=14,
            max_candidates=2,
            summarize_top=2,
            min_relevance_score=60,
        )
        repository = Repository(database_url)
        repository.init_schema()
        works = [
            {
                "id": f"https://openalex.org/W{index}",
                "display_name": title,
                "publication_date": "2026-09-01",
                "abstract_inverted_index": {"Agents": [0], word: [1]},
            }
            for index, title, word in (
                (1, "Agent memory", "remember"),
                (2, "Agent planning", "plan"),
            )
        ]
        summary = PaperSummary(
            tldr="Agents remember.",
            problem="Agent memory.",
            approach="Store memories.",
            findings=["Agents remember."],
            limitations=["No evaluation details in the abstract."],
            why_it_matters="Provides context.",
            tags=["memory"],
            evidence_basis="abstract_only",
        )
        answer = ResearchAnswer(
            answer="The memory paper describes agents that remember [1].",
            citations=[1],
            confidence="low",
            caveats=["Synthetic abstracts only."],
        )
        model = Mock()
        model.chat.completions.parse.side_effect = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            parsed=CuratorBatch(
                                decisions=[
                                    {
                                        "openalex_id": work["id"],
                                        "relevance_score": score,
                                        "reason": "Relevant",
                                    }
                                    for work, score in zip(works, (90, 80), strict=True)
                                ]
                            )
                        )
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=summary))], usage=None
            ),
            RuntimeError("Simulated interruption after the first summary"),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=summary))], usage=None
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=answer))], usage=None
            ),
        ]

        def embed_response(*, input: list[str], **_: object) -> SimpleNamespace:
            return SimpleNamespace(
                data=[
                    SimpleNamespace(embedding=[1.0, 0.0] if "memory" in text else [0.0, 1.0])
                    for text in input
                ],
                usage=None,
            )

        model.embeddings.create.side_effect = embed_response
        with (
            patch("research_radar.agents.OpenAI", return_value=model),
            patch("research_radar.openalex.httpx.Client") as openalex_http,
        ):
            openalex_http.return_value.__enter__.return_value.get.return_value = httpx.Response(
                200,
                json={"results": works},
                request=httpx.Request("GET", "https://api.openalex.org/works"),
            )
            pipeline = SyncPipeline(settings, repository, (topic,))
            with (
                self.assertLogs("research_radar.pipeline", level="ERROR"),
                self.assertRaisesRegex(RuntimeError, "Sync failed"),
            ):
                pipeline.run(force=True)
            self.assertIsNone(repository.last_success(topic.slug))
            failed_run = repository.list_runs()[0]
            self.assertEqual(failed_run["status"], "failed")
            self.assertEqual(failed_run["summarized_count"], 1)
            self.assertEqual(len(repository.pending_summaries(topic.slug, 2)), 1)

            resumed = pipeline.run(force=True)[0]
            self.assertEqual((resumed.new, resumed.indexed, resumed.summarized), (0, 0, 1))
            self.assertIsNotNone(repository.last_success(topic.slug))
            self.assertEqual(model.chat.completions.parse.call_count, 4)
            self.assertEqual(model.embeddings.create.call_count, 1)
            repeated = pipeline.run(force=True)[0]
            self.assertEqual((repeated.new, repeated.indexed, repeated.summarized), (0, 0, 0))
            self.assertEqual(model.chat.completions.parse.call_count, 4)
            self.assertEqual(model.embeddings.create.call_count, 1)
            self.assertEqual(pipeline.run()[0].status, "skipped")
            self.assertEqual(repository.dashboard_metrics()["papers"], 2)
            self.assertEqual(repository.dashboard_metrics()["summaries"], 2)

            matches = repository.search("memory", [1.0, 0.0], settings.embedding_model_id)
            self.assertEqual(matches[0]["title"], "Agent memory")
            self.assertGreater(matches[0]["semantic_score"], matches[1]["semantic_score"])
            self.assertGreater(matches[0]["lexical_score"], matches[1]["lexical_score"])
            self.assertEqual(len(repository.list_papers(query="memory")), 1)
            self.assertEqual(repository.search("memory", [1.0, 0.0], "other:model"), [])

            with (
                patch.object(web, "repository", repository),
                patch.object(web, "settings", settings),
                patch.object(web, "topics", (topic,)),
                patch.object(web, "topics_by_slug", {topic.slug: topic}),
                TestClient(web.app) as client,
            ):
                for path in ("/", "/runs", "/ask", f"/papers/{matches[0]['id']}"):
                    self.assertEqual(client.get(path).status_code, 200)
                self.assertEqual(client.get("/health").json(), {"status": "ok"})
                response = client.post("/ask", data={"question": "What do agents remember?"})
                self.assertEqual(response.status_code, 200)
                self.assertIn(answer.answer, response.text)
                self.assertIn("Experimental", response.text)
                self.assertIn('class="source-item cited"', response.text)

                model.chat.completions.parse.side_effect = None
                for invalid_text, invalid_ids, error in (
                    ("Invented source [99].", [99], "unavailable source"),
                    ("Malformed reference [1, 2].", [1, 2], "malformed citations"),
                    ("Unclosed reference [1.", [1], "malformed citations"),
                    ("Wrong source [2].", [1], "do not match"),
                ):
                    with self.subTest(answer=invalid_text):
                        invalid = answer.model_copy(
                            update={
                                "answer": invalid_text,
                                "citations": invalid_ids,
                            }
                        )
                        model.chat.completions.parse.return_value = SimpleNamespace(
                            choices=[SimpleNamespace(message=SimpleNamespace(parsed=invalid))],
                            usage=None,
                        )
                        response = client.post("/ask", data={"question": "Agent memory?"})
                        self.assertIn(error, response.text)
                        self.assertNotIn('class="answer-card"', response.text)

                with patch.object(settings, "enable_public_ask", False):
                    calls = model.embeddings.create.call_count
                    response = client.post("/ask", data={"question": "Agent memory?"})
                    self.assertIn("disabled", response.text)
                    self.assertEqual(model.embeddings.create.call_count, calls)


if __name__ == "__main__":
    unittest.main()
