from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from research_radar.models import Topic, Usage


class Settings(BaseSettings):
    database_url: str = "postgresql://radar:radar@localhost:5432/radar"
    ai_provider: Literal["openai", "ollama"] = "ollama"
    ai_curation_batch_size: int = Field(default=8, ge=1, le=50)
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen3:8b"
    ollama_embedding_model: str = "embeddinggemma"
    openalex_api_key: str | None = None
    openalex_email: str = "research-radar@example.com"
    topics_path: Path = Path("topics.toml")
    enable_public_ask: bool = False
    source_repository_url: str | None = None
    openai_input_cost_per_million_usd: float = 0.75
    openai_output_cost_per_million_usd: float = 4.50
    embedding_cost_per_million_usd: float = 0.02

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def require_openai_key(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when AI_PROVIDER=openai")
        return self.openai_api_key

    @property
    def ai_api_key(self) -> str:
        return "ollama" if self.ai_provider == "ollama" else self.require_openai_key()

    @property
    def ai_base_url(self) -> str | None:
        return self.ollama_base_url if self.ai_provider == "ollama" else None

    @property
    def ai_model(self) -> str:
        return self.ollama_model if self.ai_provider == "ollama" else self.openai_model

    @property
    def ai_embedding_model(self) -> str:
        if self.ai_provider == "ollama":
            return self.ollama_embedding_model
        return self.openai_embedding_model

    @property
    def embedding_model_id(self) -> str:
        return f"{self.ai_provider}:{self.ai_embedding_model}"

    def estimate_cost(self, usage: Usage) -> float:
        if self.ai_provider == "ollama":
            return 0
        return (
            usage.input_tokens * self.openai_input_cost_per_million_usd
            + usage.output_tokens * self.openai_output_cost_per_million_usd
            + usage.embedding_tokens * self.embedding_cost_per_million_usd
        ) / 1_000_000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def load_topics() -> tuple[Topic, ...]:
    path = get_settings().topics_path
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
        topics = tuple(Topic.model_validate(item) for item in data.get("topics", []))
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise RuntimeError(f"Unable to load topic configuration from {path}: {exc}") from exc

    if not topics:
        raise RuntimeError(f"No topics are configured in {path}")
    slugs = [topic.slug for topic in topics]
    if len(slugs) != len(set(slugs)):
        raise RuntimeError(f"Topic slugs must be unique in {path}")
    return topics
