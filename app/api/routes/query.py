"""Query endpoints (Phase 6)."""

from __future__ import annotations

import logging
import os

import chromadb
from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatRequest, ChatResponse, CollectionListResponse, MessageResponse
from app.embeddings.embedding_factory import EmbeddingFactory
from app.generation.rag_chain import get_rag_chain
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.sparse_retriever import BM25Retriever

__all__ = ["router"]

router = APIRouter(tags=["Query"])
logger = logging.getLogger(__name__)


def _fallback_answer(question: str, docs: list) -> str:
    if not docs:
        return (
            "I could not find relevant context in the indexed collection to answer this question."
        )

    excerpt = (docs[0].page_content or "").strip().replace("\n", " ")
    excerpt = " ".join(excerpt.split())
    if len(excerpt) > 500:
        excerpt = f"{excerpt[:500].rstrip()}..."

    return (
        "LLM generation is currently unavailable, so this is an extractive summary "
        f"from the top retrieved chunk for '{question}':\n\n{excerpt}"
    )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run a RAG query against a specified collection and retriever type."""
    try:
        emb = EmbeddingFactory.get("minilm")
        chroma_host = os.environ.get("CHROMA_HOST", "localhost")
        chroma_port = int(os.environ.get("CHROMA_PORT", "8000"))

        dense = DenseRetriever(
            embedding_model=emb,
            collection_name=request.collection_name,
            chroma_host=chroma_host,
            chroma_port=chroma_port,
        )

        if request.retriever_type == "dense":

            def retrieve_docs(question: str):
                return dense.retrieve(question, k=request.k)

        elif request.retriever_type == "sparse":
            sparse = BM25Retriever()

            def retrieve_docs(question: str):
                return sparse.retrieve(question, k=request.k)

        elif request.retriever_type == "hybrid":
            sparse = BM25Retriever()
            hybrid = HybridRetriever(dense, sparse)

            def retrieve_docs(question: str):
                return hybrid.retrieve(question, k=request.k)

        else:
            raise ValueError(f"Unknown retriever type: {request.retriever_type}")

        chain = get_rag_chain(retrieve_docs)

        # We need the sources alongside the answer.
        # But get_rag_chain takes retriever | format_docs.
        # To get sources, we must manually fetch them or have the chain return them.
        # Since get_rag_chain uses StrOutputParser, we manually retrieve first.
        docs = retrieve_docs(request.question)
        try:
            answer = chain.invoke(request.question)
        except Exception:  # noqa: BLE001 - fallback must handle model/runtime failures
            logger.exception("LLM generation failed; returning extractive fallback answer")
            answer = _fallback_answer(request.question, docs)

        sources = [{"page_content": d.page_content, **(d.metadata or {})} for d in docs]
        return ChatResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.exception("Chat query failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/collections", response_model=CollectionListResponse)
def list_collections() -> CollectionListResponse:
    """List all available ChromaDB collections."""
    try:
        chroma_host = os.environ.get("CHROMA_HOST", "localhost")
        chroma_port = int(os.environ.get("CHROMA_PORT", "8000"))
        client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        collections = client.list_collections()
        # Chroma can return objects or strings depending on client/server versions.
        names = [c.name if hasattr(c, "name") else str(c) for c in collections]
        return CollectionListResponse(collections=names)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/collection/{name}", response_model=MessageResponse)
def delete_collection(name: str) -> MessageResponse:
    """Delete a specific ChromaDB collection."""
    try:
        chroma_host = os.environ.get("CHROMA_HOST", "localhost")
        chroma_port = int(os.environ.get("CHROMA_PORT", "8000"))
        client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        client.delete_collection(name)
        return MessageResponse(message=f"Collection {name} deleted successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
