"""Prompt templates (Phase 4)."""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

__all__ = ["CONDENSE_PROMPT", "RAG_PROMPT"]

RAG_PROMPT = PromptTemplate.from_template(
    "You are an academic research assistant. "
    "Answer the following question based ONLY on the provided context.\n"
    "If the context does not contain the answer, "
    "politely state that you cannot answer based on the given text.\n"
    "Cite the sources for your statements.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:",
)
CONDENSE_PROMPT = PromptTemplate.from_template(
    "Given the following conversation and a follow up question, "
    "rephrase the follow up question to be a standalone question.\n\n"
    "Chat History:\n{chat_history}\n"
    "Follow Up Input: {question}\n"
    "Standalone question:",
)
