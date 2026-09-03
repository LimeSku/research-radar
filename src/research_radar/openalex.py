from __future__ import annotations

from datetime import date
from typing import Any

import httpx

from research_radar.config import Settings
from research_radar.models import DiscoveredPaper, Topic

OPENALEX_URL = "https://api.openalex.org/works"
OPENALEX_FIELDS = ",".join(
    (
        "id",
        "doi",
        "display_name",
        "abstract_inverted_index",
        "authorships",
        "primary_location",
        "best_oa_location",
        "publication_date",
        "cited_by_count",
    )
)


class OpenAlexClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def discover(self, topic: Topic, from_date: date) -> list[DiscoveredPaper]:
        params = {
            "search": topic.query,
            "filter": f"from_publication_date:{from_date.isoformat()},has_abstract:true",
            "sort": "publication_date:desc",
            "per_page": str(topic.max_candidates),
            "select": OPENALEX_FIELDS,
        }
        if self._settings.openalex_api_key:
            params["api_key"] = self._settings.openalex_api_key

        headers = {
            "User-Agent": f"ResearchRadar/0.1 (mailto:{self._settings.openalex_email})"
        }
        with httpx.Client(timeout=30, headers=headers) as client:
            response = client.get(OPENALEX_URL, params=params)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text[:300].replace("\n", " ")
            raise RuntimeError(
                f"OpenAlex returned HTTP {response.status_code}: {detail}"
            ) from exc

        payload = response.json()
        return [self._parse_work(work) for work in payload.get("results", [])]

    @staticmethod
    def _parse_work(work: dict[str, Any]) -> DiscoveredPaper:
        openalex_id = str(work["id"])
        publication_date = work.get("publication_date")
        if not publication_date:
            raise RuntimeError(f"OpenAlex work {openalex_id} has no publication date")

        primary_location = work.get("primary_location") or {}
        best_oa_location = work.get("best_oa_location") or {}
        source = primary_location.get("source") or {}
        authors = [
            authorship["author"]["display_name"]
            for authorship in work.get("authorships", [])
            if authorship.get("author", {}).get("display_name")
        ]

        return DiscoveredPaper(
            openalex_id=openalex_id,
            doi=work.get("doi"),
            title=str(work["display_name"]),
            abstract=reconstruct_abstract(work.get("abstract_inverted_index") or {}),
            authors=authors,
            venue=source.get("display_name"),
            published_at=date.fromisoformat(str(publication_date)),
            source_url=primary_location.get("landing_page_url") or work.get("doi") or openalex_id,
            pdf_url=best_oa_location.get("pdf_url"),
            oa_license=best_oa_location.get("license"),
            cited_by_count=int(work.get("cited_by_count") or 0),
        )


def reconstruct_abstract(index: dict[str, list[int]]) -> str:
    if not index:
        return ""
    positions = [(position, word) for word, offsets in index.items() for position in offsets]
    return " ".join(word for _, word in sorted(positions))
