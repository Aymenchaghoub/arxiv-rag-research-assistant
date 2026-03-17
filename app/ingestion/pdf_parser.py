"""PDF parsing utilities.

Uses PyMuPDF (`fitz`) for fast text extraction and `tiktoken` for token counts.
Section extraction is best-effort based on common paper headings.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import tiktoken

__all__ = ["PDFParseResult", "PDFParser"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PDFParseResult:
    """Output of PDF parsing."""

    full_text: str
    pages: list[str]
    sections: dict[str, str]
    page_count: int
    token_count: int


class PDFParser:
    """Parse PDFs into raw text + lightweight section splits."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    def parse(self, pdf_path: str) -> PDFParseResult:
        """Parse a PDF path.

        Args:
            pdf_path: Path to a PDF file.

        Returns:
            Parsed result with full text, extracted sections, page and token counts.
        """

        path = Path(pdf_path)
        try:
            doc = fitz.open(str(path))
        except FileNotFoundError:
            logger.warning("PDF not found", extra={"pdf_path": str(path)})
            return PDFParseResult(full_text="", pages=[], sections={}, page_count=0, token_count=0)
        except Exception as exc:  # noqa: BLE001 - fitz can raise many types
            logger.warning("Failed to open PDF", extra={"pdf_path": str(path), "error": str(exc)})
            return PDFParseResult(full_text="", pages=[], sections={}, page_count=0, token_count=0)

        try:
            pages: list[str] = []
            for page in doc:
                pages.append(page.get_text("text") or "")
            full_text = "\n".join(pages).strip()
            page_count = doc.page_count
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to extract PDF text", extra={"pdf_path": str(path), "error": str(exc)}
            )
            return PDFParseResult(full_text="", pages=[], sections={}, page_count=0, token_count=0)
        finally:
            doc.close()

        token_count = len(self._encoding.encode(full_text)) if full_text else 0
        sections = self._extract_sections(full_text)

        return PDFParseResult(
            full_text=full_text,
            pages=pages,
            sections=sections,
            page_count=page_count,
            token_count=token_count,
        )

    def _extract_sections(self, text: str) -> dict[str, str]:
        if not text.strip():
            return {}

        # Normalize whitespace a bit to help matching.
        normalized = re.sub(r"[ \t]+", " ", text)

        headings = [
            ("Abstract", r"(?mi)^\s*abstract\s*$"),
            ("Introduction", r"(?mi)^\s*(1\s*[\.\)]\s*)?introduction\s*$"),
            ("Method", r"(?mi)^\s*(method|methods|methodology)\s*$"),
            ("Results", r"(?mi)^\s*(results|experiments|evaluation)\s*$"),
            ("Conclusion", r"(?mi)^\s*(conclusion|conclusions)\s*$"),
        ]

        # Find heading positions.
        hits: list[tuple[int, str]] = []
        for name, pattern in headings:
            m = re.search(pattern, normalized)
            if m:
                hits.append((m.start(), name))
        if not hits:
            return {"FullText": text}

        hits.sort(key=lambda x: x[0])
        sections: dict[str, str] = {}
        for i, (start, name) in enumerate(hits):
            end = hits[i + 1][0] if i + 1 < len(hits) else len(normalized)
            chunk = normalized[start:end].strip()
            # Remove the heading line itself.
            chunk = re.sub(
                r"(?mi)^\s*[^\n]*\b" + re.escape(name) + r"\b[^\n]*\n", "", chunk, count=1
            )
            sections[name] = chunk.strip()

        return {k: v for k, v in sections.items() if v}
