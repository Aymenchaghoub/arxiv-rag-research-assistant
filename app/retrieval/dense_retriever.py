"""Dense retrieval via ChromaDB (HTTP client mode).

This module talks directly to a running Chroma server (e.g. in Docker) using
`chromadb.HttpClient`.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Protocol

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document

__all__ = ["DenseRetriever"]

logger = logging.getLogger(__name__)


class _EmbeddingLike(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class DenseRetriever:
    """ChromaDB-backed dense retriever."""

    def __init__(
        self,
        embedding_model: _EmbeddingLike,
        collection_name: str = "papers",
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
    ) -> None:
        self._embeddings = embedding_model
        self._collection_name = collection_name
        self._client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        self._collection: Collection | None = None

    def index_documents(self, chunks: list[Document]) -> int:
        """Upsert documents into a Chroma collection.

        Args:
            chunks: LangChain Documents to index.

        Returns:
            Number of documents upserted.
        """

        if not chunks:
            return 0

        collection = self._get_or_create_collection()

        texts = [c.page_content for c in chunks]
        metadatas = [self._sanitize_metadata(c.metadata or {}) for c in chunks]
        ids = [self._doc_id(c) for c in chunks]

        start = time.perf_counter()
        embeddings = self._embeddings.embed_documents(texts)
        embed_s = time.perf_counter() - start

        start = time.perf_counter()
        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        upsert_s = time.perf_counter() - start

        logger.info(
            "Indexed documents in Chroma",
            extra={
                "collection": self._collection_name,
                "count": len(chunks),
                "embed_seconds": embed_s,
                "upsert_seconds": upsert_s,
            },
        )

        return len(chunks)

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        """Retrieve top-k relevant documents for a query."""

        if not query.strip():
            return []

        collection = self._get_or_create_collection()
        q_emb = self._embeddings.embed_query(query)

        start = time.perf_counter()
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        seconds = time.perf_counter() - start

        docs: list[Document] = []
        docs_list = (res.get("documents") or [[]])[0]
        metas_list = (res.get("metadatas") or [[]])[0]
        dists_list = (res.get("distances") or [[]])[0]

        for text, meta, dist in zip(docs_list, metas_list, dists_list, strict=False):
            md = dict(meta or {})
            md["dense_distance"] = dist
            md["retriever"] = "dense"
            docs.append(Document(page_content=text or "", metadata=md))

        logger.info(
            "Dense retrieval complete",
            extra={
                "k": k,
                "returned": len(docs),
                "seconds": seconds,
                "collection": self._collection_name,
            },
        )
        return docs

    def delete_collection(self) -> None:
        """Delete the active Chroma collection (useful for experiments)."""

        try:
            self._client.delete_collection(name=self._collection_name)
            self._collection = None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to delete collection",
                extra={"collection": self._collection_name, "error": str(exc)},
            )

    def _get_or_create_collection(self) -> Collection:
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(name=self._collection_name)
        return self._collection

    @staticmethod
    def _doc_id(doc: Document) -> str:
        if doc.metadata and "doc_id" in doc.metadata and str(doc.metadata["doc_id"]).strip():
            return str(doc.metadata["doc_id"])
        return str(uuid.uuid4())

    @staticmethod
    def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
        # Chroma metadata values must be scalars, not arrays/objects.
        out: dict[str, Any] = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
            elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                out[k] = ", ".join(v)
            else:
                out[k] = str(v)
        return out
