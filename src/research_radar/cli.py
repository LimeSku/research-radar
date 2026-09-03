from __future__ import annotations

import argparse
import logging

import uvicorn

from research_radar.config import get_settings, load_topics
from research_radar.database import Repository
from research_radar.pipeline import SyncPipeline
from research_radar.selfcheck import run as run_selfcheck


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radar", description="Monitor and understand scientific literature."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="Create the database schema.")

    sync_parser = subparsers.add_parser("sync", help="Run due monitoring topics.")
    sync_parser.add_argument("--topic", help="Run one topic by slug.")
    sync_parser.add_argument(
        "--force", action="store_true", help="Ignore configured topic cadences."
    )

    serve_parser = subparsers.add_parser("serve", help="Start the web application.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")

    subparsers.add_parser("self-check", help="Run dependency-free core checks.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "self-check":
        run_selfcheck()
        print("Self-check passed.")
        return

    if args.command == "serve":
        uvicorn.run(
            "research_radar.web:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return

    settings = get_settings()
    repository = Repository(settings.database_url)
    repository.init_schema()

    if args.command == "init-db":
        print("Database schema is ready.")
        return

    pipeline = SyncPipeline(settings, repository, load_topics())
    for result in pipeline.run(topic_slug=args.topic, force=args.force):
        print(
            f"{result.topic_slug}: {result.status}; fetched={result.fetched}, "
            f"new={result.new}, indexed={result.indexed}, summarized={result.summarized}, "
            f"estimated_cost=${result.estimated_cost_usd:.4f}"
        )


if __name__ == "__main__":
    main()
