"""ArXiv API loader.

Fetches papers using the official `arxiv` client, downloads PDFs, and writes
metadata to disk for reproducibility.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import arxiv

__all__ = ["ArxivLoader", "PaperMetadata"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PaperMetadata:
    """Serializable metadata for a single ArXiv paper."""

    title: str
    authors: list[str]
    abstract: str
    arxiv_id: str
    published_date: str
    pdf_url: str
    pdf_path: str | None = None


class ArxivLoader:
    """Fetch papers from ArXiv and download PDFs."""

    def __init__(self, raw_dir: str = "data/raw") -> None:
        self._raw_dir = Path(raw_dir)
        self._raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch_papers(self, query: str, max_results: int = 100) -> list[dict[str, Any]]:
        """Fetch papers and download PDFs to `data/raw/`.

        Args:
            query: ArXiv search query.
            max_results: Maximum number of results to fetch.

        Returns:
            List of metadata dicts (one per successfully processed paper).
        """

        if not query.strip():
            query = "large language models NLP 2024 2025"

        logger.info(
            "Fetching papers from ArXiv",
            extra={"query": query, "max_results": max_results},
        )

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)

        papers: list[PaperMetadata] = []
        for idx, result in enumerate(client.results(search), start=1):
            meta = self._result_to_metadata(result)

            pdf_path = self._download_with_retries(result=result, arxiv_id=meta.arxiv_id)
            meta = PaperMetadata(
                **{
                    **asdict(meta),
                    "pdf_path": str(pdf_path) if pdf_path else None,
                }
            )

            if meta.pdf_path is None:
                logger.warning(
                    "Skipping paper: PDF download failed",
                    extra={"arxiv_id": meta.arxiv_id, "title": meta.title},
                )
                continue

            papers.append(meta)

            if idx % 10 == 0:
                logger.info("Progress", extra={"downloaded": len(papers), "seen": idx})

        metadata_path = self._write_metadata(papers=papers, query=query, max_results=max_results)
        logger.info(
            "Finished ArXiv ingestion",
            extra={"papers_downloaded": len(papers), "metadata_path": str(metadata_path)},
        )

        return [asdict(p) for p in papers]

    def _result_to_metadata(self, result: arxiv.Result) -> PaperMetadata:
        published = result.published
        published_iso = published.astimezone(UTC).isoformat() if published else ""

        return PaperMetadata(
            title=(result.title or "").strip(),
            authors=[a.name for a in (result.authors or [])],
            abstract=(result.summary or "").strip(),
            arxiv_id=(
                result.get_short_id() if hasattr(result, "get_short_id") else result.entry_id
            ).strip(),
            published_date=published_iso,
            pdf_url=str(result.pdf_url),
        )

    def _download_with_retries(self, result: arxiv.Result, arxiv_id: str) -> Path | None:
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                # Ensure stable filename.
                target = self._raw_dir / f"{arxiv_id}.pdf"
                if target.exists():
                    return target

                tmp_path = result.download_pdf(dirpath=str(self._raw_dir))
                tmp_path = Path(tmp_path)

                if tmp_path != target:
                    try:
                        tmp_path.replace(target)
                    except OSError:
                        # If rename fails (e.g., cross-device), keep original name.
                        return tmp_path
                return target
            except Exception as exc:  # noqa: BLE001 - external library exceptions vary
                wait_s = 2.0 * attempt
                logger.warning(
                    "PDF download failed; retrying",
                    extra={
                        "arxiv_id": arxiv_id,
                        "attempt": attempt,
                        "wait_s": wait_s,
                        "error": str(exc),
                    },
                )
                time.sleep(wait_s)

        return None

    def _write_metadata(self, papers: list[PaperMetadata], query: str, max_results: int) -> Path:
        ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = self._raw_dir / f"metadata_{ts}.json"
        payload = {
            "query": query,
            "max_results": max_results,
            "papers": [asdict(p) for p in papers],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path
