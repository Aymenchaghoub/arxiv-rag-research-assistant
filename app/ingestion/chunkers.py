"""Chunking strategies.

Implements 6 chunking strategies for benchmarking and production ingestion.
Each chunk is returned as a LangChain `Document` with required metadata.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

import nltk
import tiktoken
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.pdf_parser import PDFParseResult

__all__ = [
    "BaseChunker",
    "ChunkerFactory",
    "ChunkInput",
    "PageLevelChunker",
    "RecursiveCharacterChunker",
    "SectionBasedChunker",
    "SemanticChunker",
    "SentenceChunker",
    "TokenBasedChunker",
]

logger = logging.getLogger(__name__)

ChunkStrategyName = Literal[
    "recursive",
    "sentence",
    "page",
    "semantic",
    "section_based",
    "token",
]


@dataclass(frozen=True, slots=True)
class ChunkInput:
    """Input required by chunkers for consistent metadata."""

    arxiv_id: str
    title: str
    authors: list[str]
    parsed: PDFParseResult


class BaseChunker(abc.ABC):
    """Abstract base class for all chunkers."""

    def __init__(self, *, strategy: str, encoding_name: str = "cl100k_base") -> None:
        self._strategy = strategy
        self._encoding = tiktoken.get_encoding(encoding_name)

    @property
    def strategy(self) -> str:
        return self._strategy

    @abc.abstractmethod
    def chunk(self, item: ChunkInput) -> list[Document]:
        """Chunk a parsed paper into LangChain Documents."""

    def _token_count(self, text: str) -> int:
        return len(self._encoding.encode(text)) if text else 0

    def _base_metadata(self, item: ChunkInput) -> dict[str, Any]:
        return {
            "arxiv_id": item.arxiv_id,
            "title": item.title,
            "authors": item.authors,
            "section": None,
            "page": None,
            "strategy": self.strategy,
        }

    def _finalize(self, *, docs: list[Document]) -> list[Document]:
        for i, d in enumerate(docs):
            d.metadata = {
                **(d.metadata or {}),
                "chunk_index": i,
                "token_count": self._token_count(d.page_content),
            }
        return docs


class RecursiveCharacterChunker(BaseChunker):
    """LangChain RecursiveCharacterTextSplitter chunker."""

    def __init__(self, *, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        super().__init__(strategy="recursive")
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    def chunk(self, item: ChunkInput) -> list[Document]:
        texts = self._splitter.split_text(item.parsed.full_text)
        docs = [
            Document(page_content=t, metadata=self._base_metadata(item)) for t in texts if t.strip()
        ]
        return self._finalize(docs=docs)


class SentenceChunker(BaseChunker):
    """Chunker grouping N sentences per chunk using NLTK."""

    def __init__(self, *, sentences_per_chunk: int = 5) -> None:
        super().__init__(strategy="sentence")
        self._sentences_per_chunk = max(1, sentences_per_chunk)
        self._ensure_nltk_punkt()

    def chunk(self, item: ChunkInput) -> list[Document]:
        sentences = nltk.sent_tokenize(item.parsed.full_text)
        chunks: list[str] = []
        for i in range(0, len(sentences), self._sentences_per_chunk):
            chunk = " ".join(sentences[i : i + self._sentences_per_chunk]).strip()
            if chunk:
                chunks.append(chunk)

        docs = [Document(page_content=t, metadata=self._base_metadata(item)) for t in chunks]
        return self._finalize(docs=docs)

    def _ensure_nltk_punkt(self) -> None:
        start = time.perf_counter()
        downloaded: list[str] = []

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
            downloaded.append("punkt")

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            downloaded.append("punkt_tab")

        if downloaded:
            logger.info(
                "Downloaded NLTK tokenizers",
                extra={"resources": downloaded, "seconds": time.perf_counter() - start},
            )


class PageLevelChunker(BaseChunker):
    """One chunk per PDF page."""

    def __init__(self) -> None:
        super().__init__(strategy="page")

    def chunk(self, item: ChunkInput) -> list[Document]:
        docs: list[Document] = []
        for page_idx, text in enumerate(item.parsed.pages, start=1):
            t = (text or "").strip()
            if not t:
                continue
            meta = {**self._base_metadata(item), "page": page_idx}
            docs.append(Document(page_content=t, metadata=meta))
        return self._finalize(docs=docs)


class SemanticChunker(BaseChunker):
    """Semantic chunking using LangChain SemanticChunker."""

    def __init__(
        self,
        *,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        breakpoint_threshold_type: str = "percentile",
    ) -> None:
        super().__init__(strategy="semantic")
        start = time.perf_counter()
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self._splitter = LCSemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
        )
        logger.info(
            "Loaded SemanticChunker embeddings",
            extra={"model": embedding_model_name, "seconds": time.perf_counter() - start},
        )

    def chunk(self, item: ChunkInput) -> list[Document]:
        docs = self._splitter.create_documents([item.parsed.full_text])
        for d in docs:
            d.metadata = {**self._base_metadata(item), **(d.metadata or {})}
        return self._finalize(docs=docs)


class SectionBasedChunker(BaseChunker):
    """One chunk per section with recursive sub-chunking for long sections."""

    def __init__(
        self, *, subchunk_token_limit: int = 1000, chunk_size: int = 512, chunk_overlap: int = 50
    ) -> None:
        super().__init__(strategy="section_based")
        self._subchunk_token_limit = subchunk_token_limit
        self._recursive = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    def chunk(self, item: ChunkInput) -> list[Document]:
        if not item.parsed.sections:
            meta = {**self._base_metadata(item), "section": "FullText"}
            docs = [Document(page_content=item.parsed.full_text, metadata=meta)]
            return self._finalize(docs=docs)

        docs: list[Document] = []
        for section_name, section_text in item.parsed.sections.items():
            st = (section_text or "").strip()
            if not st:
                continue
            if self._token_count(st) > self._subchunk_token_limit:
                parts = self._recursive.split_text(st)
                for part in parts:
                    if part.strip():
                        meta = {**self._base_metadata(item), "section": section_name}
                        docs.append(Document(page_content=part, metadata=meta))
            else:
                meta = {**self._base_metadata(item), "section": section_name}
                docs.append(Document(page_content=st, metadata=meta))

        return self._finalize(docs=docs)


class TokenBasedChunker(BaseChunker):
    """Split by exact token count using tiktoken."""

    def __init__(self, *, chunk_size: int = 400, overlap: int = 50) -> None:
        super().__init__(strategy="token")
        self._chunk_size = max(1, chunk_size)
        self._overlap = max(0, min(overlap, self._chunk_size - 1))

    def chunk(self, item: ChunkInput) -> list[Document]:
        tokens = self._encoding.encode(item.parsed.full_text)
        docs: list[Document] = []

        start = 0
        while start < len(tokens):
            end = min(len(tokens), start + self._chunk_size)
            text = self._encoding.decode(tokens[start:end]).strip()
            if text:
                docs.append(Document(page_content=text, metadata=self._base_metadata(item)))
            if end == len(tokens):
                break
            start = end - self._overlap

        return self._finalize(docs=docs)


class ChunkerFactory:
    """Factory for chunkers used in benchmarking."""

    @staticmethod
    def get(strategy: ChunkStrategyName, **kwargs: Any) -> BaseChunker:
        """Build a chunker instance by strategy name."""

        if strategy == "recursive":
            return RecursiveCharacterChunker(**kwargs)
        if strategy == "sentence":
            return SentenceChunker(**kwargs)
        if strategy == "page":
            return PageLevelChunker()
        if strategy == "semantic":
            return SemanticChunker(**kwargs)
        if strategy == "section_based":
            return SectionBasedChunker(**kwargs)
        if strategy == "token":
            return TokenBasedChunker(**kwargs)

        raise ValueError(f"Unknown chunk strategy: {strategy}")
