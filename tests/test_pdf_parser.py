from __future__ import annotations

from pathlib import Path

import fitz

from app.ingestion.pdf_parser import PDFParser


def _write_minimal_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Abstract\nWe propose a method.\n\nIntroduction\nHello world.")
    doc.save(str(path))
    doc.close()


def test_pdf_parser_extracts_text_pages_and_tokens(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _write_minimal_pdf(pdf_path)

    parser = PDFParser()
    result = parser.parse(str(pdf_path))

    assert result.page_count == 1
    assert result.full_text
    assert result.pages and len(result.pages) == 1
    assert result.token_count > 0
    assert "Abstract" in result.sections


def test_pdf_parser_missing_file_is_graceful(tmp_path: Path) -> None:
    parser = PDFParser()
    result = parser.parse(str(tmp_path / "missing.pdf"))
    assert result.page_count == 0
    assert result.full_text == ""
