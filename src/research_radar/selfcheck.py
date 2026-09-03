import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from research_radar.agents import AgentService
from research_radar.config import Settings
from research_radar.models import CuratorBatch, CuratorDecision, DiscoveredPaper, Topic, Usage
from research_radar.openalex import reconstruct_abstract
from research_radar.pipeline import topic_is_due


def run() -> None:
    assert reconstruct_abstract({"world": [1], "Hello": [0]}) == "Hello world"

    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    assert topic_is_due(None, 6, now)
    assert topic_is_due(now - timedelta(hours=6), 6, now)
    assert not topic_is_due(now - timedelta(hours=5), 6, now)

    settings = Settings(
        _env_file=None,
        ai_provider="openai",
        openai_input_cost_per_million_usd=1,
        openai_output_cost_per_million_usd=2,
        embedding_cost_per_million_usd=0.5,
    )
    assert settings.estimate_cost(Usage(1_000_000, 1_000_000, 1_000_000)) == 3.5
    assert settings.ai_model == "gpt-5.4-mini"
    assert settings.embedding_model_id == "openai:text-embedding-3-small"

    ollama_settings = Settings(
        _env_file=None,
        ai_provider="ollama",
        ollama_model="qwen3:8b",
        ollama_embedding_model="embeddinggemma",
    )
    assert ollama_settings.ai_api_key == "ollama"
    assert ollama_settings.ai_model == "qwen3:8b"
    assert ollama_settings.embedding_model_id == "ollama:embeddinggemma"
    assert ollama_settings.estimate_cost(Usage(1_000, 1_000, 1_000)) == 0

    batch_sizes: list[int] = []

    class FakeCompletions:
        def parse(self, **kwargs: object) -> SimpleNamespace:
            messages = kwargs["messages"]
            assert isinstance(messages, list)
            payload = json.loads(messages[1]["content"])
            batch_sizes.append(len(payload["papers"]))
            parsed = CuratorBatch(
                decisions=[
                    CuratorDecision(
                        openalex_id=paper["openalex_id"], relevance_score=80, reason="Relevant"
                    )
                    for paper in payload["papers"]
                ]
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))], usage=None
            )

    agent = AgentService.__new__(AgentService)
    agent._settings = Settings(
        _env_file=None,
        ai_provider="ollama",
        ai_curation_batch_size=2,
        ollama_model="qwen3:8b",
    )
    agent._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    topic = Topic(
        slug="agentic-ai",
        name="Agentic AI",
        description="Planning and tool use.",
        query="agents",
        cadence_hours=6,
        lookback_days=14,
        max_candidates=10,
        summarize_top=2,
        min_relevance_score=60,
    )
    papers = [
        DiscoveredPaper(
            openalex_id=f"W{index}",
            doi=None,
            title=f"Paper {index}",
            abstract="An agent uses tools.",
            authors=[],
            venue=None,
            published_at=now.date(),
            source_url="https://example.org",
            pdf_url=None,
            oa_license=None,
        )
        for index in range(5)
    ]
    decisions, _ = agent.curate(topic, papers)
    assert len(decisions) == 5
    assert batch_sizes == [2, 2, 1]
