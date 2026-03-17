from __future__ import annotations

from app.ingestion.chunkers import ChunkerFactory, ChunkInput
from app.ingestion.pdf_parser import PDFParseResult


def _parsed_sample() -> PDFParseResult:
    text = "\n".join(
        [
            "Abstract",
            "We propose a method.",
            "Introduction",
            "This is the introduction. It has multiple sentences.",
            "Method",
            "We use transformers.",
            "Conclusion",
            "It works.",
        ]
    )
    return PDFParseResult(
        full_text=text,
        pages=[text],
        sections={
            "Abstract": "We propose a method.",
            "Introduction": "This is the introduction. It has multiple sentences.",
            "Method": "We use transformers.",
            "Conclusion": "It works.",
        },
        page_count=1,
        token_count=10,
    )


def test_recursive_chunker_metadata() -> None:
    chunker = ChunkerFactory.get("recursive", chunk_size=40, chunk_overlap=5)
    docs = chunker.chunk(
        ChunkInput(arxiv_id="1234.5678", title="T", authors=["A"], parsed=_parsed_sample())
    )
    assert docs
    for d in docs:
        assert d.metadata["arxiv_id"] == "1234.5678"
        assert d.metadata["title"] == "T"
        assert d.metadata["strategy"] == "recursive"
        assert "chunk_index" in d.metadata
        assert "token_count" in d.metadata


def test_section_based_chunker_includes_section() -> None:
    chunker = ChunkerFactory.get("section_based", subchunk_token_limit=1000)
    docs = chunker.chunk(
        ChunkInput(arxiv_id="1234.5678", title="T", authors=["A"], parsed=_parsed_sample())
    )
    assert {d.metadata["section"] for d in docs} >= {
        "Abstract",
        "Introduction",
        "Method",
        "Conclusion",
    }


def test_page_level_chunker_sets_page() -> None:
    chunker = ChunkerFactory.get("page")
    docs = chunker.chunk(
        ChunkInput(arxiv_id="1234.5678", title="T", authors=["A"], parsed=_parsed_sample())
    )
    assert docs
    assert all(d.metadata["page"] == 1 for d in docs)


def test_token_chunker_produces_documents() -> None:
    chunker = ChunkerFactory.get("token", chunk_size=10, overlap=2)
    docs = chunker.chunk(
        ChunkInput(arxiv_id="1234.5678", title="T", authors=["A"], parsed=_parsed_sample())
    )
    assert docs
