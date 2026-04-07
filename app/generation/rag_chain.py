"""RAGChain (Phase 4)."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSerializable
from langchain_huggingface import HuggingFaceEndpoint

from app.generation.prompt_templates import RAG_PROMPT

__all__ = ["get_rag_chain", "format_docs"]


def format_docs(docs: list[Any]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain(retriever: Any) -> RunnableSerializable:
    """Builds the LCEL RAG Chain: retriever | format_docs | prompt | llm | StrOutputParser."""
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        max_new_tokens=512,
        temperature=0.1,
    )

    rag_chain = (
        {
            "context": RunnableLambda(retriever) | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain
