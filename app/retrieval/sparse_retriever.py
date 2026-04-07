"""Sparse retrieval via BM25.

Uses `rank-bm25` with a simple whitespace tokenization.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

__all__ = ["BM25Index", "BM25Retriever"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BM25Index:
    """Serializable BM25 index data."""

    tokenized_corpus: list[list[str]]
    documents: list[Document]


class BM25Retriever:
    """BM25 sparse retriever with optional on-disk persistence."""

    def __init__(self, *, index_path: str = "data/processed/bm25_index.pkl") -> None:
        self._index_path = Path(index_path)
        self._bm25: BM25Okapi | None = None
        self._docs: list[Document] = []
        self._tokenized: list[list[str]] = []

    def index_documents(self, chunks: list[Document]) -> int:
        """Build BM25 index from provided chunks and persist it."""

        self._docs = chunks[:]
        self._tokenized = [self._tokenize(d.page_content) for d in self._docs]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        self._persist()
        logger.info(
            "Built BM25 index",
            extra={"count": len(self._docs), "path": str(self._index_path)},
        )
        return len(self._docs)

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        """Retrieve top-k documents with BM25 scores in metadata."""

        if not query.strip():
            return []

        if self._bm25 is None:
            self._load_if_exists()
        if self._bm25 is None:
            return []

        q_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(q_tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        out: list[Document] = []
        for idx, score in ranked:
            src = self._docs[idx]
            md: dict[str, Any] = dict(src.metadata or {})
            md["bm25_score"] = float(score)
            md["retriever"] = "sparse"
            out.append(Document(page_content=src.page_content, metadata=md))

        return out

    def _persist(self) -> None:
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            payload = BM25Index(tokenized_corpus=self._tokenized, documents=self._docs)
            self._index_path.write_bytes(pickle.dumps(payload))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to persist BM25 index",
                extra={"path": str(self._index_path), "error": str(exc)},
            )

    def _load_if_exists(self) -> None:
        try:
            if not self._index_path.exists():
                return
            payload = pickle.loads(self._index_path.read_bytes())
            if not isinstance(payload, BM25Index):
                return
            self._docs = payload.documents
            self._tokenized = payload.tokenized_corpus
            self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        except FileNotFoundError:
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load BM25 index",
                extra={"path": str(self._index_path), "error": str(exc)},
            )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in (text or "").lower().split() if t]
