"""API request/response schemas (Phase 6)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "CollectionListResponse",
    "IngestRequest",
    "IngestResponse",
    "MessageResponse",
]


class IngestRequest(BaseModel):
    arxiv_id: str
    chunking_strategy: str
    chunk_size: int


class IngestResponse(BaseModel):
    document_count: int


class ChatRequest(BaseModel):
    question: str
    collection_name: str
    retriever_type: str
    k: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


class CollectionListResponse(BaseModel):
    collections: list[str]


class MessageResponse(BaseModel):
    message: str
