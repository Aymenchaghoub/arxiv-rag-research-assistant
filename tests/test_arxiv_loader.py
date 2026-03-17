from __future__ import annotations

from pathlib import Path

import arxiv

from app.ingestion.arxiv_loader import ArxivLoader


class _FakeAuthor:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeResult:
    def __init__(self, *, arxiv_id: str, raw_dir: Path) -> None:
        self._arxiv_id = arxiv_id
        self.title = f"Title {arxiv_id}"
        self.authors = [_FakeAuthor("Alice"), _FakeAuthor("Bob")]
        self.summary = "Abstract text."
        self.entry_id = f"http://arxiv.org/abs/{arxiv_id}"
        self.pdf_url = f"http://arxiv.org/pdf/{arxiv_id}.pdf"
        self.published = None
        self._raw_dir = raw_dir

    def get_short_id(self) -> str:
        return self._arxiv_id

    def download_pdf(self, dirpath: str) -> str:
        p = Path(dirpath) / f"{self._arxiv_id}.pdf"
        p.write_bytes(b"%PDF-1.4 fake\n")
        return str(p)


def test_arxiv_loader_writes_metadata_and_pdfs(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"

    fake_results = [_FakeResult(arxiv_id="1234.5678", raw_dir=raw_dir)]

    def _fake_results(self, search: arxiv.Search):  # noqa: ARG001
        return iter(fake_results)

    monkeypatch.setattr(arxiv.Client, "results", _fake_results)

    loader = ArxivLoader(raw_dir=str(raw_dir))
    out = loader.fetch_papers(query="test query", max_results=1)

    assert out and out[0]["arxiv_id"] == "1234.5678"
    assert (raw_dir / "1234.5678.pdf").exists()
    assert list(raw_dir.glob("metadata_*.json"))
