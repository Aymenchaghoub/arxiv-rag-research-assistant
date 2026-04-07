from __future__ import annotations

import math
import os
import uuid
from pathlib import Path

import pytest
from langchain_core.documents import Document


def test_retrievers_placeholder() -> None:
    # Phase 3 will replace this with real retriever tests.
    assert True


class _FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> list[float]:
        # Tiny deterministic vector: [len, vowels, spaces]
        vowels = sum(1 for c in text.lower() if c in "aeiou")
        spaces = text.count(" ")
        return [float(len(text)), float(vowels), float(spaces)]


@pytest.mark.integration
def test_dense_retriever_indexes_and_queries_against_chroma() -> None:
    from app.retrieval.dense_retriever import DenseRetriever

    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "8000"))
    collection = f"test_{uuid.uuid4().hex}"

    try:
        retriever = DenseRetriever(
            embedding_model=_FakeEmbeddings(),
            collection_name=collection,
            chroma_host=host,
            chroma_port=port,
        )
    except ValueError as exc:
        pytest.skip(f"Chroma not reachable at {host}:{port} ({exc})")
    try:
        docs = [
            Document(
                page_content="transformers for nlp",
                metadata={"doc_id": "d1", "arxiv_id": "x"},
            ),
            Document(page_content="graph algorithms", metadata={"doc_id": "d2", "arxiv_id": "y"}),
        ]
        assert retriever.index_documents(docs) == 2

        out = retriever.retrieve("nlp transformers", k=2)
        assert len(out) == 2
        assert all("dense_distance" in d.metadata for d in out)
    finally:
        retriever.delete_collection()


def test_bm25_retriever_persists_and_loads(tmp_path: Path) -> None:
    from app.retrieval.sparse_retriever import BM25Retriever

    index_path = tmp_path / "bm25.pkl"
    r1 = BM25Retriever(index_path=str(index_path))
    docs = [
        Document(page_content="large language models", metadata={"doc_id": "a"}),
        Document(page_content="support vector machines", metadata={"doc_id": "b"}),
    ]
    r1.index_documents(docs)

    r2 = BM25Retriever(index_path=str(index_path))
    out = r2.retrieve("language models", k=1)
    assert out and out[0].metadata["doc_id"] == "a"
    assert "bm25_score" in out[0].metadata


def test_hybrid_retriever_rrf_scores_are_monotonic() -> None:
    from app.retrieval.hybrid_retriever import HybridRetriever

    class _StaticRetriever:
        def __init__(self, docs: list[Document]) -> None:
            self._docs = docs

        def retrieve(self, query: str, k: int = 5) -> list[Document]:  # noqa: ARG002
            return self._docs[:k]

    d1 = Document(page_content="A", metadata={"doc_id": "1", "dense_distance": 0.1})
    d2 = Document(page_content="B", metadata={"doc_id": "2", "dense_distance": 0.2})
    s1 = Document(page_content="A", metadata={"doc_id": "1", "bm25_score": 10.0})
    s2 = Document(page_content="C", metadata={"doc_id": "3", "bm25_score": 9.0})

    hybrid = HybridRetriever(
        dense=_StaticRetriever([d1, d2]),
        sparse=_StaticRetriever([s1, s2]),
        k=60,
    )
    out = hybrid.retrieve("q", k=3)

    assert out
    assert out[0].metadata["retriever"] == "hybrid"
    assert "rrf_score" in out[0].metadata
    assert out[0].metadata["doc_id"] == "1"
    assert math.isfinite(out[0].metadata["rrf_score"])
