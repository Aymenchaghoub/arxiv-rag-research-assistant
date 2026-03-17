"""Application configuration.

All configuration is sourced from environment variables (optionally via a `.env`
file when running locally). Never hardcode secrets in code.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings for the RAG system."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Keys / observability
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    langchain_api_key: str | None = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_tracing_v2: bool = Field(default=True, alias="LANGCHAIN_TRACING_V2")

    # Vector DB
    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, alias="CHROMA_PORT")

    # RAG knobs
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    chunk_strategy: str = Field(default="recursive", alias="CHUNK_STRATEGY")
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    retriever_k: int = Field(default=5, alias="RETRIEVER_K")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")


__all__ = ["Settings"]
