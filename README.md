# Research Radar

Research Radar is a configurable research-intelligence system that discovers, evaluates,
summarizes, and indexes new scientific publications into a searchable, cited knowledge base.

It is deliberately agentic only where reasoning adds value. Collection, deduplication,
scheduling, and persistence remain deterministic; AI agents handle semantic curation,
technical summarization, and grounded synthesis.

## A real run

An Agentic AI run on September 8, 2026 collected 30 publications and produced five
structured summaries. Research Radar ran on macOS, with `qwen2.5:7b-instruct` served
by Ollama on a Windows desktop with an RTX 4070 (12 GB VRAM).
These screenshots show the populated application from that run.

![Research Radar overview showing a successful run, 30 publications, and five summaries](docs/screenshots/overview.jpg)

![Research digest showing real publications, generated briefs, and relevance scores](docs/screenshots/digest.jpg)

### Example: from abstract to research brief

**Paper:** [From Explainability to Actionability: a Tiered Adaptable Multi-Agent Framework
with Agent Reasoning Tools for Collaborative Failure Recovery](https://doi.org/10.1007/s10796-026-10815-2)
— Jie Tao and Lina Zhou, *Information Systems Frontiers*.

The Analyst Agent generated this summary from the paper's metadata and abstract:

> This study introduces a multi-agent framework designed to enhance human-AI collaboration,
> particularly in failure recovery scenarios, by leveraging explainable AI principles.
> The framework includes tiered reasoning, traceable memory for auditability, and flexible
> reasoning tools, demonstrating improvements in task performance and recovery strategies
> through empirical evaluation and multi-role user simulations.

The displayed abstract supports these main points. This is **abstract-based analysis**, not
a full-paper review or independent verification of the study's results. The abstract does
not provide numerical effect sizes or enough detail to assess broader generalizability.

The [quick start](#quick-start) uses the model demonstrated here. See
[known limitations](#known-limitations) for issues observed during the demo.

Ask is **experimental**. Citation IDs are checked against retrieved sources, but this does
not verify that a source supports a claim or that an answer is complete.

### Example: a reviewed Ask answer

On September 9, 2026, hybrid retrieval searched the 30-publication Agentic AI corpus and
returned six sources. The framework paper above ranked first. The Research Agent used
`qwen2.5:7b-instruct` with an observed 8,192-token Ollama context; query embeddings used
`embeddinggemma`. The request used 3,891 prompt tokens and 100 completion tokens.

**Question:** What mechanisms does the tiered multi-agent framework use for collaborative
failure recovery?

**Unedited model answer:**

> The tiered multi-agent framework uses tiered reasoning and flexible reasoning tools to
> adapt to failure severity and devise recovery strategies [1]. It includes a traceable
> memory bus for auditability, enabling agents to propose actionable improvement plans [1].
> These mechanisms support collaborative failure recovery by allowing agents to coordinate
> and provide informed recovery strategies based on the severity of the failure [1].

[1]: https://doi.org/10.1007/s10796-026-10815-2

**Review:** The response completed normally, and every inline citation matched source 1.
The stored OpenAlex abstract names all three mechanisms and describes diagnosis, recovery
strategies, and improvement plans. The answer compresses that description: the abstract
attributes those outcomes to the architecture as a whole, not to the memory bus alone.
This checks the answer against its abstract; it does not establish empirical effectiveness.

The model reported `high` confidence and no caveats; that is a model self-assessment, not a
calibrated reliability score. Two earlier attempts failed citation validation. The prompt
was then tightened to request a direct answer without commentary about unused sources.
This is one reviewed example, not a benchmark or a claim of consistent reliability.

## What it demonstrates

- Incremental collection of a capped sample of recent OpenAlex publications
- Structured LLM outputs with explicit Pydantic contracts
- Switchable OpenAI and local Ollama inference through one validated agent path
- Semantic curation with configurable topic-specific budgets
- Abstract-grounded summaries that avoid pretending to have read unavailable full text
- Hybrid PostgreSQL full-text and pgvector retrieval
- Experimental cited RAG answers with prompt-injection boundaries
- Idempotent jobs, resumable processing stages, run telemetry, token usage, and cost estimates
- A responsive server-rendered UI, Docker packaging, CI, scheduled jobs, and deployment IaC

## Architecture

```mermaid
flowchart LR
    A[OpenAlex API] --> B[Deterministic ingestion]
    B --> C[(PostgreSQL)]
    C --> D[Curator Agent]
    D --> E[Embedding index]
    D --> F[Analyst Agent]
    E --> G[Hybrid retrieval]
    F --> C
    G --> H[Research Agent]
    H --> I[FastAPI UI]
    C --> I
```

The hourly scheduler checks each topic's own cadence. A PostgreSQL advisory lock prevents
overlapping jobs, unique source IDs make ingestion idempotent, and explicit processing states
allow interrupted runs to continue instead of repeating completed work.

### Agent responsibilities

| Role | Input | Output |
| --- | --- | --- |
| Curator | Topic definition, titles, abstracts | Relevance score and concise rationale |
| Analyst | One selected paper's metadata and abstract | Typed research brief with findings and limitations |
| Research | A question and retrieved evidence | Grounded answer, citations, confidence, and caveats |

Discovery and indexing are intentionally not agents. Making predictable infrastructure depend
on model decisions would cost more and be harder to operate.

## Quick start

Requirements: Docker, Python 3.12+, [`uv`](https://docs.astral.sh/uv/), and optionally a free
OpenAlex API key. Choose either local Ollama (free, default) or the OpenAI API.

For the free local setup, start Ollama on the inference host (through the desktop app,
or with `ollama serve` in a separate terminal). Then download the models:

```bash
ollama pull qwen2.5:7b-instruct
ollama pull embeddinggemma
```

On the computer running Research Radar, create `.env` if it does not already exist:

```bash
cp .env.example .env
```

Then edit `.env` to use the demonstrated model and a batch size of four:

```dotenv
AI_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_EMBEDDING_MODEL=embeddinggemma
AI_CURATION_BATCH_SIZE=4
```

The project's defaults remain `qwen3:8b` and eight papers per batch; the settings above
override them. Four is the demo machine's currently configured batch size. The successful
run's exact Ollama context length was not recorded, so this is not an exact replay recipe.
Check the active context length with `ollama ps` on the inference host during a request.

If Ollama runs on another computer, also set `OLLAMA_BASE_URL` to
`http://<OLLAMA-HOST-IP>:11434/v1`, using that computer's reachable local network address.
For Ollama on the same computer, keep the default `http://localhost:11434/v1`.

Start the database and run the demonstrated topic:

```bash
docker compose up -d db
uv sync
uv run radar init-db
uv run radar sync --topic agentic-ai --force
uv run radar serve --reload
```

Open [http://localhost:8000](http://localhost:8000).

To use OpenAI instead, change these values in `.env` before starting the application:

```dotenv
AI_PROVIDER=openai
OPENAI_API_KEY=your-api-key
```

To run the complete stack in containers instead:

If Ollama runs on the same computer, set `OLLAMA_BASE_URL` in `.env` to
`http://host.docker.internal:11434/v1`: `localhost` inside the app container points to
the container itself. For another inference computer, keep its reachable network URL.
Compose respects this setting and uses `host.docker.internal` when it is unset.

```bash
docker compose up --build -d
docker compose run --rm app radar sync --force
```

## Configuration

Topics live in [`topics.toml`](topics.toml), so adding a domain does not require code changes:

```toml
[[topics]]
slug = "efficient-ml"
name = "Efficient Machine Learning"
description = "Compression, quantization, distillation, and efficient inference."
query = "efficient machine learning quantization distillation inference"
cadence_hours = 24
lookback_days = 14
max_candidates = 25
summarize_top = 5
min_relevance_score = 60
```

`max_candidates` caps ingestion and embedding work per run. `summarize_top` caps the more
expensive Analyst Agent. Every curated candidate is indexed; only the best eligible candidates
receive a full research brief.

### Collection coverage

Discovery fetches **one page of up to `max_candidates` recent publications** per topic,
sorted by publication date. It is a capped sample, not an exhaustive literature search.
The first run uses `lookback_days`; later runs start two days before the last successful
sync date. This overlap helps rediscover recent records, which are deduplicated by OpenAlex ID.

If more papers match than fit on that page, older matches can be missed as the date window
advances. Records added to OpenAlex later with older publication dates can also be missed.
`--force` bypasses the schedule; it does not paginate or backfill the corpus. Exhaustive
coverage would require pagination and a persisted collection cursor.

Environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `DATABASE_URL` | PostgreSQL connection string | Local Compose database |
| `AI_PROVIDER` | Inference backend: `ollama` or `openai` | `ollama` |
| `AI_CURATION_BATCH_SIZE` | Papers evaluated per model call | `8` |
| `OPENAI_API_KEY` | OpenAI authentication | Required only for `openai` |
| `OPENAI_MODEL` | Structured-output model | `gpt-5.4-mini` |
| `OPENAI_EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `OLLAMA_BASE_URL` | Ollama OpenAI-compatible API URL | `http://localhost:11434/v1` |
| `OLLAMA_MODEL` | Local structured-output model | `qwen3:8b` |
| `OLLAMA_EMBEDDING_MODEL` | Local embedding model | `embeddinggemma` |
| `OPENALEX_API_KEY` | Higher OpenAlex API allowance | Optional |
| `OPENALEX_EMAIL` | Contact address in the OpenAlex user agent | Placeholder |
| `TOPICS_PATH` | Topic configuration file | `topics.toml` |
| `ENABLE_PUBLIC_ASK` | Enable the model-backed Ask endpoint | `false` |
| `SOURCE_REPOSITORY_URL` | “View source” navigation link | Hidden when empty |

Ollama usage is reported as zero cost. OpenAI cost rates are configurable through
`OPENAI_INPUT_COST_PER_MILLION_USD`, `OPENAI_OUTPUT_COST_PER_MILLION_USD`, and
`EMBEDDING_COST_PER_MILLION_USD`. Update them when changing models or pricing.

## Commands

```bash
radar init-db                       # Create extensions, tables, and indexes
radar sync                          # Run topics whose cadence has elapsed
radar sync --topic agentic-ai      # Run one due topic
radar sync --topic agentic-ai --force
radar serve --reload
radar self-check                    # Run the small dependency-free invariant check
```

## Validation

```bash
uv run ruff check .
uv run radar self-check
docker compose up -d db
TEST_DATABASE_URL=postgresql://radar:radar@localhost:5432/radar \
  uv run python -m unittest discover -s tests -v
```

The integration smoke test needs PostgreSQL with pgvector and a database user allowed to
create databases. It creates a uniquely named temporary database and drops it afterward;
it does not reset the supplied database. OpenAlex and model responses are synthetic, so
the test needs no API keys, downloads, or inference server.

The test covers ingestion, deduplication, resuming after a summary failure, vector and
full-text retrieval, web pages, citation validation, and disabling Ask. CI runs the same
checks against PostgreSQL 16 with pgvector. These checks verify application behavior;
they do not measure model answer quality.

## Deployment

[`render.yaml`](render.yaml) provisions the Docker web service and a managed PostgreSQL database
in Render's Frankfurt region. Render PostgreSQL supports the pgvector extension used by the
schema.

1. Push the repository to GitHub and create a Render Blueprint from it.
2. Keep `AI_PROVIDER=openai`, then provide `OPENAI_API_KEY`, `OPENALEX_EMAIL`, and optionally
   `OPENALEX_API_KEY` when prompted.
3. Add the Render database's **external** connection string as the GitHub Actions secret
   `DATABASE_URL`.
4. Add `OPENAI_API_KEY` and optionally `OPENALEX_API_KEY` as Actions secrets, plus
   `OPENALEX_EMAIL` as an Actions variable.
5. Enable the `Research sync` workflow. It runs hourly and lets the application enforce the
   topic-specific cadences in `topics.toml`.

Keep `ENABLE_PUBLIC_ASK=false` for an unprotected public demo, or enable it only when model usage
is intentionally public. The browsing, digest, paper, and run pages remain fully usable without
exposing the model-backed endpoint.

## Data and evidence policy

Research Radar stores normalized metadata, abstracts, model outputs, and links. It does not
scrape publisher pages or treat a free-to-read URL as permission to redistribute a paper. Agent
briefs are labelled `abstract_only`, and Research Agent answers disclose that evidence boundary.

## Current scope

The first vertical slice uses one source, two interchangeable model providers, and exact vector
search. Add direct arXiv ingestion, full-text parsing for clearly licensed documents,
notifications, and approximate HNSW indexing only when corpus size or product usage justifies
them.

## Known limitations

- Generated briefs need editorial review. In the example above, the limitations field
  described missing details rather than an established limitation of the study.
- An earlier Ask trial contained malformed citation text and ended mid-sentence. The
  reviewed example above completed successfully after tightening the prompt, and malformed
  or inconsistent citations are now rejected. Completeness and factual support still need
  human review; a successful example does not establish reliability across questions.
- A fixed paper count does not guarantee that a curation batch fits the model's context.
  Qwen3.5 9B trials hit context limits or returned mismatched paper decisions; the successful
  run used `qwen2.5:7b-instruct`. Model compatibility needs validation on real inputs.

## License

[MIT](LICENSE)
