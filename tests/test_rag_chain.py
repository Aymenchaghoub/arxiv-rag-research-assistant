from __future__ import annotations

from app.generation.rag_chain import get_rag_chain


def test_rag_chain_returns_string(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    def mock_llm_invoke(*args, **kwargs):
        return "This is a mocked answer."

    # Mock HuggingFaceEndpoint directly inside rag_chain module
    monkeypatch.setattr(
        "app.generation.rag_chain.HuggingFaceEndpoint",
        lambda **kwargs: RunnableLambda(mock_llm_invoke),
    )

    def fake_retriever(query: str):
        class FakeDoc:
            page_content = "Hello context."
            metadata = {}

        return [FakeDoc()]

    chain = get_rag_chain(fake_retriever)

    # Passing just a string question
    result = chain.invoke("Hello question")
    assert isinstance(result, str)
    assert result == "This is a mocked answer."
