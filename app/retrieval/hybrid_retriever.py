"""Hybrid retrieval via Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Protocol

from langchain_core.documents import Document

__all__ = ["HybridRetriever"]


class _RetrieverLike(Protocol):
    def retrieve(self, query: str, k: int = 5) -> list[Document]: ...


class HybridRetriever:
    """Fuse dense + sparse rankings using Reciprocal Rank Fusion."""

    def __init__(self, dense: _RetrieverLike, sparse: _RetrieverLike, k: int = 60) -> None:
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = k

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        dense_docs = self._dense.retrieve(query=query, k=k)
        sparse_docs = self._sparse.retrieve(query=query, k=k)

        scores: dict[str, float] = defaultdict(float)
        payload: dict[str, dict[str, Any]] = {}

        def _key(d: Document) -> str:
            if d.metadata and "doc_id" in d.metadata:
                return str(d.metadata["doc_id"])
            # Fall back to (arxiv_id, chunk_index) if present, else content hash.
            a = str((d.metadata or {}).get("arxiv_id", ""))
            ci = str((d.metadata or {}).get("chunk_index", ""))
            if a or ci:
                return f"{a}:{ci}"
            return str(hash(d.page_content))

        for rank, d in enumerate(dense_docs):
            scores[_key(d)] += 1.0 / (rank + self._rrf_k)
            payload.setdefault(_key(d), {})["dense_distance"] = (d.metadata or {}).get(
                "dense_distance"
            )
            payload.setdefault(_key(d), {})["dense_rank"] = rank

        for rank, d in enumerate(sparse_docs):
            scores[_key(d)] += 1.0 / (rank + self._rrf_k)
            payload.setdefault(_key(d), {})["bm25_score"] = (d.metadata or {}).get("bm25_score")
            payload.setdefault(_key(d), {})["sparse_rank"] = rank

        # Prefer dense document text/metadata when available.
        by_key: dict[str, Document] = {}
        for d in sparse_docs + dense_docs:
            by_key.setdefault(_key(d), d)

        ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
        out: list[Document] = []
        for key in ranked_keys:
            base = by_key[key]
            md = dict(base.metadata or {})
            md.update(payload.get(key, {}))
            md["rrf_score"] = float(scores[key])
            md["retriever"] = "hybrid"
            out.append(Document(page_content=base.page_content, metadata=md))

        return out
