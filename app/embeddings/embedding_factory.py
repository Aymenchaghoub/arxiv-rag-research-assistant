"""Embedding model factory.

Provides a small registry of embedding backends used in the benchmark.
Models are cached in-process (loaded once, reused across calls).
"""

from __future__ import annotations

import logging
import time
from typing import ClassVar

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

__all__ = ["EmbeddingFactory"]

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """Factory for embedding models with caching."""

    _cache: ClassVar[dict[str, Embeddings]] = {}

    @classmethod
    def get(cls, model_name: str) -> Embeddings:
        """Return an embeddings instance for a given short name.

        Supported names:
        - ``bge-m3`` -> ``BAAI/bge-m3``
        - ``minilm`` -> ``all-MiniLM-L6-v2``
        - ``gte-large`` -> ``thenlper/gte-large``

        Args:
            model_name: Short model key.

        Returns:
            A LangChain Embeddings instance.
        """

        key = model_name.strip().lower()
        if key in cls._cache:
            return cls._cache[key]

        hf_name = cls._resolve_model_name(key)
        start = time.perf_counter()
        embeddings = HuggingFaceEmbeddings(model_name=hf_name)
        seconds = time.perf_counter() - start

        dim = cls._infer_dim(embeddings)
        logger.info(
            "Loaded embeddings model",
            extra={"key": key, "model": hf_name, "seconds": seconds, "dim": dim},
        )

        cls._cache[key] = embeddings
        return embeddings

    @staticmethod
    def _resolve_model_name(key: str) -> str:
        if key == "bge-m3":
            return "BAAI/bge-m3"
        if key == "minilm":
            return "all-MiniLM-L6-v2"
        if key == "gte-large":
            return "thenlper/gte-large"
        raise ValueError(f"Unknown embedding model key: {key}")

    @staticmethod
    def _infer_dim(embeddings: Embeddings) -> int:
        try:
            v = embeddings.embed_query("dimension probe")
            return len(v)
        except Exception:  # noqa: BLE001
            return -1
