"""Ingestion endpoints (Phase 6)."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException

from app.api.schemas import IngestRequest, IngestResponse
from app.embeddings.embedding_factory import EmbeddingFactory
from app.ingestion.arxiv_loader import ArxivLoader
from app.ingestion.chunkers import ChunkerFactory, ChunkInput
from app.ingestion.pdf_parser import PDFParser
from app.retrieval.dense_retriever import DenseRetriever

__all__ = ["router"]

router = APIRouter(tags=["Ingest"])
logger = logging.getLogger(__name__)


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(request: IngestRequest) -> IngestResponse:
    """Ingest a document from ArXiv, chunk it, and index it into ChromaDB."""
    try:
        loader = ArxivLoader(raw_dir="./data/raw")
        papers = loader.fetch_papers(query=request.arxiv_id, max_results=1)
        if not papers:
            raise ValueError(f"No paper found for query '{request.arxiv_id}'")

        paper = papers[0]
        pdf_path = paper.get("pdf_path")
        if not pdf_path:
            raise ValueError("PDF download failed for the given document.")

        parser = PDFParser()
        parsed = parser.parse(pdf_path)

        chunk_input = ChunkInput(
            arxiv_id=paper["arxiv_id"],
            title=paper["title"],
            authors=paper["authors"],
            parsed=parsed,
        )

        kwargs = {}
        if request.chunking_strategy in ("fixed_size", "token", "recursive"):
            kwargs["chunk_size"] = request.chunk_size

        chunker = ChunkerFactory.get(strategy=request.chunking_strategy, **kwargs)
        docs = chunker.chunk(chunk_input)

        emb = EmbeddingFactory.get("minilm")

        chroma_host = os.environ.get("CHROMA_HOST", "localhost")
        chroma_port = int(os.environ.get("CHROMA_PORT", "8000"))

        # Chroma collection names must satisfy character and length constraints.
        collection_name = request.arxiv_id.replace(".", "-").replace("/", "-")
        # ArXiv IDs can start with digits, so we prefix when needed.
        if not collection_name[0].isalpha():
            collection_name = f"arxiv-{collection_name}"

        retriever = DenseRetriever(
            embedding_model=emb,
            collection_name=collection_name,
            chroma_host=chroma_host,
            chroma_port=chroma_port,
        )
        count = retriever.index_documents(docs)

        return IngestResponse(document_count=count)
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
