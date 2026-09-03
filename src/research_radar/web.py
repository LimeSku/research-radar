from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any

import psycopg
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAIError

from research_radar.agents import AgentService
from research_radar.config import get_settings, load_topics
from research_radar.database import Repository

settings = get_settings()
topics = load_topics()
topics_by_slug = {topic.slug: topic for topic in topics}
repository = Repository(settings.database_url)
package_dir = Path(__file__).parent
templates = Jinja2Templates(directory=package_dir / "templates")


def format_date(value: date | None) -> str:
    return value.strftime("%b %d, %Y") if value else "—"


def format_datetime(value: datetime | None) -> str:
    return value.strftime("%b %d, %Y · %H:%M UTC") if value else "—"


def compact_number(value: int | None) -> str:
    number = int(value or 0)
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if number >= 1_000:
        return f"{number / 1_000:.1f}K"
    return str(number)


templates.env.filters["date"] = format_date
templates.env.filters["datetime"] = format_datetime
templates.env.filters["compact"] = compact_number


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    repository.init_schema()
    yield


app = FastAPI(
    title="Research Radar",
    description="Agentic scientific literature monitoring and research memory.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=package_dir / "static"), name="static")


def render(request: Request, template: str, **context: Any) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name=template,
        context={
            "topics": topics,
            "current_path": request.url.path,
            "ask_enabled": settings.enable_public_ask,
            "source_repository_url": settings.source_repository_url,
            "topics_by_slug": topics_by_slug,
            **context,
        },
    )


def validate_topic(topic_slug: str | None) -> str | None:
    if topic_slug is not None and topic_slug not in topics_by_slug:
        raise HTTPException(status_code=404, detail=f"Unknown topic: {topic_slug}")
    return topic_slug


@app.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    topic: Annotated[str | None, Query()] = None,
    q: Annotated[str, Query(max_length=200)] = "",
) -> HTMLResponse:
    topic_slug = validate_topic(topic)
    return render(
        request,
        "index.html",
        page_title="Digest",
        selected_topic=topic_slug,
        query=q.strip(),
        metrics=repository.dashboard_metrics(),
        papers=repository.list_papers(topic_slug=topic_slug, query=q.strip()),
        latest_run=next(iter(repository.list_runs(limit=1)), None),
    )


@app.get("/papers/{paper_id}", response_class=HTMLResponse)
def paper_detail(request: Request, paper_id: int) -> HTMLResponse:
    paper = repository.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    paper["summary_topic"] = next(
        (item for item in paper["topics"] if item["summary"]), paper["topics"][0]
    )
    return render(
        request,
        "paper.html",
        page_title=paper["title"],
        paper=paper,
    )


@app.get("/ask", response_class=HTMLResponse)
def ask_page(
    request: Request,
    topic: Annotated[str | None, Query()] = None,
) -> HTMLResponse:
    return render(
        request,
        "ask.html",
        page_title="Ask",
        selected_topic=validate_topic(topic),
        question="",
        answer=None,
        sources=[],
        error=None,
    )


@app.post("/ask", response_class=HTMLResponse)
def ask(
    request: Request,
    question: Annotated[str, Form(min_length=3, max_length=500)],
    topic: Annotated[str | None, Form()] = None,
) -> HTMLResponse:
    topic_slug = validate_topic(topic or None)
    context = {
        "page_title": "Ask",
        "selected_topic": topic_slug,
        "question": question,
        "answer": None,
        "sources": [],
        "error": None,
    }
    if not settings.enable_public_ask:
        return render(
            request,
            "ask.html",
            **{
                **context,
                "error": "Interactive Q&A is disabled on this deployment.",
            },
        )

    try:
        agent = AgentService(settings)
        vectors, _ = agent.embed([question])
        matches = repository.search(
            question,
            vectors[0],
            agent.embedding_model_id,
            topic_slug=topic_slug,
        )
        if not matches:
            return render(
                request,
                "ask.html",
                **{
                    **context,
                    "error": "No indexed publications are available for this question yet.",
                },
            )

        agent_sources = [
            {
                "citation_id": index,
                "title": source["title"],
                "abstract": source["abstract"],
                "summary": source["summary"],
            }
            for index, source in enumerate(matches, start=1)
        ]
        answer, _ = agent.answer(question, agent_sources)
        valid_citations = {
            citation for citation in answer.citations if 1 <= citation <= len(matches)
        }
        display_sources = [
            {**source, "citation_id": index, "cited": index in valid_citations}
            for index, source in enumerate(matches, start=1)
        ]
        return render(
            request,
            "ask.html",
            **{**context, "answer": answer, "sources": display_sources},
        )
    except (OpenAIError, RuntimeError) as exc:
        return render(
            request,
            "ask.html",
            **{**context, "error": f"Unable to answer: {exc}"},
        )


@app.get("/runs", response_class=HTMLResponse)
def runs(request: Request) -> HTMLResponse:
    return render(
        request,
        "runs.html",
        page_title="Runs",
        runs=repository.list_runs(),
    )


@app.get("/health")
def health() -> JSONResponse:
    try:
        healthy = repository.health()
    except psycopg.Error:
        healthy = False
    return JSONResponse(
        {"status": "ok" if healthy else "unavailable"},
        status_code=200 if healthy else 503,
    )
