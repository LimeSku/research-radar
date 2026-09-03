from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class Topic(BaseModel):
    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=300)
    query: str = Field(min_length=2, max_length=500)
    cadence_hours: int = Field(ge=1, le=720)
    lookback_days: int = Field(ge=1, le=365)
    max_candidates: int = Field(ge=1, le=100)
    summarize_top: int = Field(ge=1, le=25)
    min_relevance_score: int = Field(ge=0, le=100)


class DiscoveredPaper(BaseModel):
    openalex_id: str
    doi: str | None
    title: str
    abstract: str
    authors: list[str]
    venue: str | None
    published_at: date
    source_url: str
    pdf_url: str | None
    oa_license: str | None
    cited_by_count: int = 0


class CuratorDecision(BaseModel):
    openalex_id: str
    relevance_score: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1, max_length=500)


class CuratorBatch(BaseModel):
    decisions: list[CuratorDecision]


class PaperSummary(BaseModel):
    tldr: str = Field(min_length=1, max_length=500)
    problem: str = Field(min_length=1, max_length=1200)
    approach: str = Field(min_length=1, max_length=1200)
    findings: list[str] = Field(min_length=1, max_length=6)
    limitations: list[str] = Field(min_length=1, max_length=6)
    why_it_matters: str = Field(min_length=1, max_length=800)
    tags: list[str] = Field(min_length=1, max_length=8)
    evidence_basis: Literal["abstract_only"]


class ResearchAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=1200)
    citations: list[int]
    confidence: Literal["low", "medium", "high"]
    caveats: list[str] = Field(max_length=5)


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            embedding_tokens=self.embedding_tokens + other.embedding_tokens,
        )
